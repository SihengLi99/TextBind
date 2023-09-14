import os
import json
import base64
import hashlib
import torch
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, jsonify, send_from_directory, Blueprint
from main import parse_args
from inference import MIMPipeline
import logging
from common_utils import FileUtils
import random
random.seed(10086)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ],
)

def check_url_prefix(url_prefix):
    url_prefix = url_prefix.strip()
    if url_prefix:
        if url_prefix.startswith("/"):
            return url_prefix
        else:
            return "/" + url_prefix
    else:
        return url_prefix

PUNC = set([".", ",", "!"])

app = Flask(__name__)
args = parse_args()
url_prefix = check_url_prefix(args.url_prefix)
image_remote_path="{}/images/".format(url_prefix)
image_local_dir = os.path.abspath(os.path.dirname(__file__)) +  image_remote_path
FileUtils.check_dirs(image_local_dir)
logging.info("Remote image path: {}".format(image_remote_path))
logging.info("Local image dir: {}".format(image_local_dir))
api = Blueprint('api', __name__, url_prefix=url_prefix if url_prefix else None)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info("Loading model for demo...")
agent = MIMPipeline(args, device)
engines = agent.engines
examples = FileUtils.load_file(args.demo_example_path) if args.demo_example_path else []

def run_inference(data, selection):
    global agent
    return agent.run(data, selection)

def process_user_input(user_input):
    soup = BeautifulSoup(user_input, 'html.parser')
    images = soup.find_all('img')

    for img in images:
        if not img['src'].startswith(image_remote_path):
            sha1 = hashlib.sha1()
            img_data = img['src'].split(',')[1]
            img_bytes = base64.b64decode(img_data)
            sha1.update(img_bytes)
            img_ext = img['src'].split(',')[0].split('/')[1].split(';')[0]
            img_filename = "{}.{}".format(sha1.hexdigest(), img_ext)
            image_local_path = os.path.join(image_local_dir, img_filename)
            if not os.path.exists(image_local_path):
                with open(image_local_path, 'wb') as f:
                    f.write(img_bytes)
            user_input = user_input.replace(img['src'], os.path.join(image_remote_path, img_filename))
    return user_input


def parse_chat_history(chat_history_html):
    # Parse the chat history using BeautifulSoup
    soup = BeautifulSoup(chat_history_html, 'html.parser')
    messages = soup.find_all(class_='message')
    chat_history = []
    for message in messages:
        role = 'user' if 'user-message' in message['class'] else 'assistant'
        cur_message = {"role": role, "data": []}

        if role == 'user':
            if message:
                user_input = process_user_input(str(message))
                for element in BeautifulSoup(user_input, 'html.parser').recursiveChildGenerator():
                    if element.name == 'img':
                        message_data = {'type': 'image', 'value': element['src']}
                        cur_message['data'].append(message_data)
                    elif element.name is None:
                        message_data = {'type': 'text', 'value': element.string.strip()}
                        cur_message['data'].append(message_data)
        else:
            for child in message.children:
                if child.name == 'img':
                    message_data = {
                        'type': 'image',
                        'value': child['src'],
                    }
                    cur_message['data'].append(message_data)
                elif child.name == 'p':
                    message_data = {
                        'type': 'text',
                        'value': child.get_text(strip=True),
                    }
                    cur_message['data'].append(message_data)
                elif child.name == 'div':
                    for element in BeautifulSoup(str(child), 'html.parser').recursiveChildGenerator():
                        if element.name == 'img':
                            message_data = {'type': 'image', 'value': element['src']}
                            cur_message['data'].append(message_data)
                        elif element.name is None:
                            message_data = {'type': 'text', 'value': element.string.strip()}
                            cur_message['data'].append(message_data)
                else:
                    continue

        if cur_message['data']:
            chat_history.append(cur_message)
    return chat_history
    

@api.route('/')
def index():
    return render_template('index.html', url_prefix=url_prefix)


@api.route('/random_conversation', methods=['GET'])
def random_dialogue():
    # Implement a function to fetch a random dialogue history from the server
    if examples:
        ex = random.choice(examples)
        for item in ex['conversation']:
            if item['image_list'] and not item['image_list'][0].startswith(image_remote_path):
                item['image_list'] = ["{}/{}".format(image_remote_path, m) for m in item['image_list']]
    else:
        ex = [{'type': 'text', 'value': "No examples on server"}]
    logging.info("Example: \n{}".format(json.dumps(ex)))
    return jsonify(ex)


@api.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(image_local_dir, filename)


def prepare_model_input(chat_history):
    data = {"conversation": []}
    for msg in chat_history:
        mesg_data, image_list = [], []
        for it in msg['data']:
            if it['type'] == "text":
                mesg_data.append(it['value'])
            elif it['type'] == "image":
                mesg_data.append("<image>")
                image_list.append(it['value'].split("/")[-1])
        # mesg_data = " ".join(mesg_data)
        mesg_data = [it for it in mesg_data if it]
        mesg_data_str = mesg_data[0]
        for m in mesg_data[1:]:
            if m[0] in PUNC:
                mesg_data_str += m
            else:
                mesg_data_str += " " + m
        data["conversation"].append(
            {'role': "user" if msg['role'] == "user" else "assistant", "content": mesg_data_str, "image_list": image_list, "caption_list": []}
        )
    return data


def count_images_and_words(data):
    n_words = 0
    n_images = 0
    for turn in data['conversation']:
        n_images += len(turn['image_list'])
        n_words += len(turn['content'].split())
    return n_words, n_images


def split_model_output(gen_text, use_image_id):
    if not use_image_id:
        gen_text_splits = gen_text.split("<image>")
        return gen_text_splits
    else:
        gen_text_splits, image_order = [], []
        gen_text_words = gen_text.split()
        cache = []
        for wi, w in enumerate(gen_text_words):
            if w.startswith("<image"):
                gen_text_splits.append(" ".join(cache))
                cache = []
                image_order.append(int(w[len("<image"):-1]))
            else:
                cache.append(w)
        if cache:
            gen_text_splits.append(" ".join(cache))
        return gen_text_splits, image_order
            

def parse_model_output(out, turn_id=-1):
    response = []
    gen_text, gen_imgs = out['conversation'][turn_id]['content'], out['conversation'][turn_id]['image_list']
    # gen_text_splits = gen_text.split("<image>")
    if "<image>" not in gen_text:
        gen_text_splits, image_order = split_model_output(gen_text, True)
        gen_imgs = [gen_imgs[i] for i in image_order]
    else:
        gen_text_splits = split_model_output(gen_text, False)
    n_splits = len(gen_text_splits)
    assert len(gen_imgs) == (n_splits-1)
    for i in range(n_splits):
        response.append(
            {"type": "text", "value": gen_text_splits[i].replace("</s>", "").strip()}
        )
        if i != (n_splits-1):
            response.append(
                {"type": "image", "value": "{}/{}".format(image_remote_path, gen_imgs[i])}
            )
    return response

@api.route('/engine-list', methods=['GET'])
def engine_list():
    global engines
    if not engines:
        el = [
                {'id': '---', 'name': '---'},
        ]
    else:
        el = engines
    return jsonify(el)

@api.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get('user_input', '')
    nlp_engine = data.get('nlp_engine', 's2')
    # logging.info("Selected Engine: {}".format(nlp_engine))
    chat_history_html = data.get('chat_history', '')
    chat_history = parse_chat_history(chat_history_html) if chat_history_html else []

    # try:
    if user_input:
        # Process the user_input with your NLP engine here and get the response
        model_input = prepare_model_input(chat_history)
        logging.info("Model Input: \n{}".format(json.dumps(model_input)))
        n_words, n_images = count_images_and_words(model_input)
        logging.info("{} words and {} images in conversation".format(n_words, n_images))
        if n_words >= args.safe_word_num or n_images >= args.safe_image_num:
            response_items = [{'type': 'text', 'value': "I'm sorry that I may not be able to continue this conversation, due to my limited GPU memory. The admin set the safe number of words and images in conversation to {} and {}, respectively. This strategy is to avoid the core dump of the GPU. If you want to experience longer conversation, please run our model and code on your more powerful GPUs!".format(args.safe_word_num, args.safe_image_num)}]
        else:
            model_output = run_inference(model_input, nlp_engine)
            logging.info("Model Response: \n{}".format(json.dumps(model_output)))
            response_items = parse_model_output(model_output)
    else:
        response_items = [{'type': 'text', 'value': "Please input some text or images."}]
    # except Exception as e:
    #     logging.info("Error message:\ne")
    #     response_items = [{'type': 'text', 'value': "Our server met some errors... Contact us to report the bug."}]

    return jsonify({'response': response_items})


app.register_blueprint(api)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=args.port)
