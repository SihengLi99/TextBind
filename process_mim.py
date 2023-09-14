import os
import re
import json
import requests
from PIL import Image, ImageOps
from io import BytesIO
import fire
from common_utils import logger, FileUtils, MPUtils
from common_utils import PrepUtils
import constaints as C


def clean_mim_data(data_path, mim_edit_dist=0.1, min_turn_num=3):
    cleaned = []
    data = FileUtils.load_file(data_path)
    logger.info("{} exemples loaded".format(len(data)))
    visited = set()
    stats = {
        "repeat_image": 0, "unseen_image": 0, "no_image": 0, "repeat_response": 0,
        "invalid_turns": 0, "wrong_role": 0, "changed_caption": 0, "too_short": 0
    }
    for ex in data:
        if PrepUtils.has_repeated_images(ex):
            stats['repeat_image'] += 1
            continue
        if PrepUtils.has_unseen_image(ex):
            stats['unseen_image'] += 1
            continue
        if not PrepUtils.has_image(ex):
            stats['no_image'] += 1
            continue
        response, caption_list = ex['response'], ex['image']
        if response in visited:
            stats['repeat_response'] += 1
            continue
        else:
            visited.add(response)
        try:
            turns = PrepUtils.split_turn(response)
        except ValueError:
            stats['wrong_role'] += 1
            continue
        if not PrepUtils.is_valid_turn(turns):
            stats['invalid_turns'] += 1
            continue
        if len(turns) < min_turn_num:
            stats['too_short'] += 1
            continue
        turns = [PrepUtils.remove_non_paired_img_tag(t) for t in turns]
        new_turns, is_valid =  [], True
        for sent in turns:
            caption_matched, has_caption = True, False
            for i, _ in enumerate(caption_list):
                gen_cap = PrepUtils.extract_text(sent, i)
                if gen_cap is None:
                    continue
                has_caption = True
                clean_cap = PrepUtils.clean_tag(caption_list[i])
                ed_score = PrepUtils.edit_distance(gen_cap, clean_cap) / len(clean_cap)
                if ed_score > mim_edit_dist:
                    caption_matched = False
            if has_caption and not caption_matched:
                is_valid = False
                stats['changed_caption'] += 1
                break
            new_turns.append(sent)
        if is_valid:
            ex['response'] = "\n\n".join(new_turns)
            cleaned.append(ex)
    logger.info("{} exemples left after cleaning".format(len(cleaned)))
    FileUtils.save_file(cleaned, FileUtils.handle_file_extension(data_path, "clean", "add") , 'json')
    stats = ["{}\t{:.2f}%\t{}/{}".format(k, v / len(data) * 100, v, len(data)) for k, v in stats.items()]
    logger.info("Statistical results: \n{}".format("\n".join(stats)))


def prepare_model_input(data_path, image_dir="./data/mim_images", force_downloading=False, nproc=16):
    corpus = FileUtils.load_file(data_path, 'json')
    logger.info("Loaded {} instances".format(len(corpus)))
    img_data = PrepUtils.gather_image_data(corpus)
    faied_image_urls = []
    if force_downloading:
        FileUtils.check_dirs(image_dir)
        img_data_shards = MPUtils.prepare_shards(img_data, nproc)
        args_list = [(img_data_shards[i], image_dir, i) for i in range(nproc)]
        MPUtils.mp_func(download_images, args_list)
    logger.info("Finished downloading")
    for proc_id in range(nproc):
        fpath = image_dir + "/failed.proc{}.txt".format(proc_id)
        if FileUtils.exists(fpath):
            faied_image_urls += FileUtils.load_file(fpath)
    faied_image_urls = set(faied_image_urls)
    new_corpus, stats = [], {"incomplete_image": 0}
    for idx, data in enumerate(corpus):
        image_path_list = ["{}.png".format(mi) for mi in data["image_idx"]]
        has_failed_images = False
        if faied_image_urls:
            for u in data["url"]:
                if u in faied_image_urls:
                    has_failed_images = True
                    break
            if has_failed_images:
                stats['incomplete_image'] += 1
                continue
        else:
            if not PrepUtils.check_image_list(image_dir, image_path_list):
                stats['incomplete_image'] += 1
                continue
        conversation = []
        for turn in data["response"].split("\n\n"):
            image_ids = PrepUtils.extract_idx(turn)
            image_list = [image_path_list[j] for j in image_ids]
            url_list = [data["url"][j] for j in image_ids]
            caption_list = [PrepUtils.clean_tag(data["image"][j]) for j in image_ids]
            turn = PrepUtils.sub_image_tag(turn)
            if turn.startswith(C.ASSISTANT):
                conversation.append({"role": "assistant", "content": turn[len(C.ASSISTANT):].strip(), "image_list":image_list, "caption_list": caption_list, "url_list": url_list})
            elif turn.startswith(C.HUMAN):
                conversation.append({"role": "user", "content": turn[len(C.HUMAN):].strip(), "image_list":image_list, "caption_list": caption_list, "url_list": url_list})
        new_corpus.append({"conversation": conversation, 'image_dir': image_dir})
    logger.info("{} instances left after cleaning".format(len(new_corpus)))
    FileUtils.save_file(new_corpus, FileUtils.handle_file_extension(data_path, "reform", "add"), 'json')
    FileUtils.save_file(img_data, FileUtils.handle_file_extension(data_path, "img-cap", "add"), 'json')
    stats = ["{}\t{:.2f}%\t{}/{}".format(k, v / len(corpus) * 100, v, len(corpus)) for k, v in stats.items()]
    logger.info("Statistical results: \n{}".format("\n".join(stats)))


def get_image_caption(corpus_path, save_path):
    logger.info("Start processing...")
    corpus = json.load(open(corpus_path, "r"))
    logger.info(f"load {len(corpus)} instances")
    new_corpus = []
    for data in corpus:
        for image_idx, caption in zip(data["image_idx"], data["image"]):
            caption = re.findall(r'<img\d+>(.*?)<\/img\d+>', caption)[0]
            new_corpus.append({
                "image": f"{image_idx}.png",
                "caption": caption.strip()
            })
    json.dump(new_corpus, open(save_path, "w"), indent=4)


def download_images(image_data, image_dir, proc_id, max_try_num=5, image_size=512):
    failed_data = []
    if isinstance(image_data, str):
        image_data = FileUtils.load_file(image_data)
    FileUtils.check_dirs(image_dir)
    logger.info("Proc-{} | Downloading images for shard with {} examples".format(proc_id, len(image_data)))
    for idx in range(len(image_data)):
        image_basename, url = image_data[idx]['image'], image_data[idx]['url']
        image_path = "{}/{}".format(image_dir, image_basename)
        if not PrepUtils.check_image_file(image_path):     
            logger.info("Proc-{} | Downloading from {} for {}".format(proc_id, url, image_basename))
            try_num = 0
            while try_num < max_try_num:
                try:
                    response = requests.get(url, headers=C.HEADERS)
                    image = Image.open(BytesIO(response.content)).convert("RGB")
                    image = PrepUtils.resize_and_pad(image, (image_size, image_size))
                    image.save(image_path)
                    break
                except:
                    try_num += 1
            if try_num >= max_try_num:
                failed_data.append(url)
                logger.info("Failed to download {}".format(url))                            
    FileUtils.save_file(failed_data, image_dir + "/failed.proc{}.txt".format(proc_id))


def split_data(data_path, save_prefix, valid_num=100, test_num=100):
    import random
    random.seed(10086)
    data = FileUtils.load_file(data_path)
    ids = list(range(len(data)))
    random.shuffle(ids)
    data = [data[i] for i in ids]
    valid = data[:valid_num]
    test = data[valid_num:valid_num+test_num]
    train = data[valid_num+test_num:]
    FileUtils.save_file(train, save_prefix + ".train.json")
    FileUtils.save_file(valid, save_prefix + ".valid.json")
    FileUtils.save_file(test, save_prefix + ".test.json")


def data_statistics(data_path):
    from sacremoses import MosesTokenizer
    from collections import Counter
    from tqdm import tqdm
    tokenizer = MosesTokenizer(lang='en')
    mean = lambda x: sum(x) / len(x)

    def compute_div_score(ngram_counters):
        div_score_turns = []
        for k, cs in ngram_counters.items():
            div_score = 0
            for n in range(2, 5):
                total_num = sum(cs[n].values())
                unique_num = len(cs[n])
                div_score += unique_num / total_num
            div_score_turns.append(div_score)
        return mean(div_score_turns)

    def traditional_statistics(data):
        total_ex_num = len(data)
        conversation_lens, instruct_lens, response_lens = [], [], []
        conversation_image_nums, instruct_image_nums, response_image_nums = [], [], []
        turn_nums = []
        for ex in data:
            conversation = ex['conversation']
            turn_nums.append(len(conversation) / 2)
            ci, ct, ri, rt, ii, it = 0, 0, 0, 0, 0, 0
            for c in conversation:
                if c['role'] == "user":
                    ci += len(c['image_list'])
                    ii += len(c['image_list'])
                    content = tokenizer.tokenize(c['content'], escape=False)
                    ct += len(content)
                    it += len(content)
                elif c['role'] == "assistant":
                    ci += len(c['image_list'])
                    ri += len(c['image_list'])
                    content = tokenizer.tokenize(c['content'], escape=False)
                    ct += len(content)
                    rt += len(content)
                else:
                    raise ValueError(c['role'])
            conversation_lens.append(ct)
            conversation_image_nums.append(ci)
            instruct_lens.append(it)
            instruct_image_nums.append(ii)
            response_image_nums.append(ri)
            response_lens.append(rt)
        logger.info("total_ex_num: {}".format(total_ex_num))
        logger.info("turn_nums: {}".format(mean(turn_nums)))
        logger.info("conversation_lens: {}".format(mean(conversation_lens)))
        logger.info("instruct_lens: {}".format(mean(instruct_lens)))
        logger.info("response_lens: {}".format(mean(response_lens)))
        logger.info("conversation_image_nums: {}".format(mean(conversation_image_nums)))
        logger.info("instruct_image_nums: {}".format(mean(instruct_image_nums)))
        logger.info("response_image_nums: {}".format(mean(response_image_nums)))

    def image_diversity(data):
        user_image_nums, assist_image_nums = dict(), dict()
        for ex in data:
            conversation = ex['conversation']
            for cidx, c in enumerate(conversation):
                if c['role'] == "user":
                    cidx = cidx // 2
                    if cidx in user_image_nums:
                        user_image_nums[cidx].append(len(c['image_list']))
                    else:
                        user_image_nums[cidx] = [len(c['image_list'])]
                elif c['role'] == "assistant":
                    cidx = cidx // 2
                    if cidx in assist_image_nums:
                        assist_image_nums[cidx].append(len(c['image_list']))
                    else:
                        assist_image_nums[cidx] = [len(c['image_list'])]
                else:
                    raise ValueError(c['role'])
        for i in range(len(user_image_nums)):
            logger.info("turn: {}\tuser_image_num: {}".format(i, mean(user_image_nums[i])))
        for i in range(len(assist_image_nums)):
            logger.info("turn: {}\tassistant_image_num: {}".format(i, mean(assist_image_nums[i])))

    def text_diversity(data):
        ngram_counters = dict() 
        user_ngram_counters = dict()
        assitant_ngram_counters = dict()
        for ex in tqdm(data):
            conversation = ex['conversation']
            for cidx, c in enumerate(conversation):
                cidx = cidx // 2
                if cidx not in ngram_counters:
                    ngram_counters[cidx] = {2: Counter(), 3: Counter(), 4: Counter()}
                if cidx not in user_ngram_counters:
                    user_ngram_counters[cidx] = {2: Counter(), 3: Counter(), 4: Counter()}
                if cidx not in assitant_ngram_counters:
                    assitant_ngram_counters[cidx] = {2: Counter(), 3: Counter(), 4: Counter()}
                content = tokenizer.tokenize(c['content'], escape=False)
                if c['role'] == "user":
                    for n in range(2, 5):
                        nragms = PrepUtils.extract_ngrams(content, n=n)
                        user_ngram_counters[cidx][n].update(nragms)
                        ngram_counters[cidx][n].update(nragms)
                elif c['role'] == "assistant":
                    for n in range(2, 5):
                        nragms = PrepUtils.extract_ngrams(content, n=n)
                        assitant_ngram_counters[cidx][n].update(nragms)                    
                        ngram_counters[cidx][n].update(nragms)                    
        logger.info("Turns: {}".format(list(ngram_counters.keys())))
        logger.info("Overall Div score: {}".format(compute_div_score(ngram_counters)))
        logger.info("User Div score: {}".format(compute_div_score(user_ngram_counters)))
        logger.info("Assistant Div score: {}".format(compute_div_score(assitant_ngram_counters)))

    data = FileUtils.load_file(data_path)
    logger.info("----------------------- traditional_statistics -----------------------")
    # traditional_statistics(data)
    logger.info("----------------------- text_diversity -----------------------")
    text_diversity(data)
    logger.info("----------------------- image_diversity -----------------------")
    # image_diversity(data)


def analyze_human_annotation(data_path="./annotation.csv", has_header=True):
    from collections import Counter
    ann = FileUtils.load_file(data_path)
    if has_header:
        ann = ann[1:]
    counters = {"quality": Counter(), "character": Counter(), "error": Counter()}
    n = 0
    for row in ann:
        quality = row[2]
        if quality:
            n += 1
            counters['quality'].update([quality])
            if quality == "Poor":
                counters['error'].update([it.strip() for it in row[4].split(',')])
            else:
                counters['character'].update([it.strip() for it in row[3].split(',')])
    for k, v in counters.items():
        for label, freq in v.most_common():
            logger.info("{} | {}: {}".format(k, label, freq / n))


if __name__ == "__main__":
    fire.Fire({
        "prepare_model_input": prepare_model_input,
        "clean_mim_data": clean_mim_data,
        "split_data": split_data,
        "download_images": download_images,
        "data_statistics": data_statistics,
        "analyze_human_annotation": analyze_human_annotation
    })