import torch
import json
import os
import yaml
import os.path as path
import tarfile
from glob import glob
from io import BytesIO
import csv
import multiprocessing as mp
from PIL import Image, ImageOps
import re
import math
import logging
import constaints as C

logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
handler.setLevel(logging.INFO)
# Add auto-flushing to the StreamHandler
handler.terminator = '\n'
handler.flush = lambda: handler.stream.flush()
logger.addHandler(handler)



class FileType:
    PT = "pt"
    TXT = "txt"
    JSON = "json"
    TSV = "tsv"
    TAR = "tar"
    YAML = "yaml"
    CSV = "csv"
    ALL = ["pt", "txt", "json", "tsv", "tar", "yaml", "csv"]


class FileExtensionType:
    ADD = "add"
    CHANGE = "change"


LANGUAGE_LIST = ["en", "de", "zh", "vi", "fr"]


class PrepUtils:

    @staticmethod
    def extract_ngrams(sent, n, return_index=False):
        if isinstance(sent, str):
            sent = sent.split()
        ngrams = [tuple(sent[i:i + n]) for i in range(len(sent) - n + 1)]
        if not return_index:
            return ngrams
        else:
            ids = list(range(len(sent)))
            ngram_ids = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
            return ngrams, ngram_ids

    @staticmethod
    def resize_and_pad(image, size=(512, 512)):
        aspect_ratio = float(image.width) / float(image.height)

        if aspect_ratio > 1:
            new_width = size[0]
            new_height = int(size[0] / aspect_ratio)
        else:
            new_height = size[1]
            new_width = int(size[1] * aspect_ratio)

        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        padded_image = ImageOps.expand(resized_image, (
            (size[0] - new_width) // 2,
            (size[1] - new_height) // 2,
            (size[0] - new_width + 1) // 2,
            (size[1] - new_height + 1) // 2), fill='white')
        
        return padded_image
    
    @staticmethod
    def gather_image_data(corpus):
        img_dict = dict()
        for data in corpus:
            for idx, url, cap in zip(data["image_idx"], data["url"], data['image']):
                cap = PrepUtils.clean_tag(cap)
                if idx not in img_dict:
                    img_dict[idx] = (url, cap)
        ret = [{"image": "{}.png".format(idx), "url": v[0], "caption": v[1]} for idx, v in img_dict.items()]
        return ret
    
    @staticmethod
    def check_image_file(image_path):
        if not FileUtils.exists(image_path):
            return False
        else:
            try:
                _ = Image.open(image_path)
                return True
            except Exception:
                logger.info("Find damaged image {}".format(image_path))
                return False
    
    @staticmethod
    def check_image_list(image_dir, image_path_list):
        for image_path in image_path_list:
            if not PrepUtils.check_image_file("{}/{}".format(image_dir, image_path)):
                return False
        return True
    
    @staticmethod
    def edit_distance(s1, s2):
        m, n = len(s1), len(s2)
        dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0:
                    dp[i][j] = j
                elif j == 0:
                    dp[i][j] = i
                elif s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

        return dp[m][n]

    @staticmethod
    def extract_text(text, x):
        pattern = f"<img{x}>.*?</img{x}>"
        match = re.search(pattern, text)
        if match:
            original_text = match.group(0)[len(f"<img{x}>"):-len(f"</img{x}>")].strip()
        else:
            original_text = None
        return original_text

    @staticmethod
    def extract_idx(text):
        image_idx_list = []
        matches = re.findall(r'<img\d>.*?</img\d>', text)
        for match in matches:
            image_idx = re.findall(r"<img(\d)>", match)[0]
            image_idx_list.append(int(image_idx))
        return image_idx_list
    
    @staticmethod
    def sub_image_tag(text):
        return re.sub(r'<img\d>.*?</img\d>', "<image>", text)

    @staticmethod
    def split_turn(dialogue):
        start_index = dialogue.index(C.HUMAN)
        sents = dialogue[start_index:].strip().split("\n")
        turns = []
        for sent in sents:
            sent = sent.strip()
            if sent.startswith(C.HUMAN):
                turns.append(sent)
            elif sent.startswith(C.ASSISTANT):
                turns.append(sent)
            else:
                if sent:
                    turns[-1] += "\n" + sent
        return turns

    @staticmethod
    def is_valid_turn(turns):
        for i, t in enumerate(turns):
            if C.HUMAN in t and C.ASSISTANT in t:
                return False
            if t.startswith(C.HUMAN) and (i % 2) == 0:
                continue
            if t.startswith(C.ASSISTANT) and (i % 2) == 1:
                continue
            return False
        return True

    @staticmethod
    def has_repeated_images(ex):
        img_tags = ["<img{}>".format(i) for i in range(len(ex['image']))]
        response = ex['response']
        for it in img_tags:
            n = len(StringUtils.find_all_indices(response, it))
            if n > 1:
                return True
        return False

    @staticmethod
    def has_unseen_image(ex):
        n_image = len(ex['image'])
        pattern = r"<img(\d+)>"
        img_ids = [int(it) for it in re.findall(pattern, ex['response'])]
        for i in img_ids:
            if i >= n_image:
                return True
        return False

    @staticmethod
    def has_image(ex):
        pattern = r'<img(\d+)>.*?</img\1>'
        return bool(re.search(pattern, ex['response']))

    @staticmethod
    def remove_non_paired_img_tag(text):
        img_tags = re.findall(r'<img\d+>', text)
        # Iterate through the tags and remove them if there is no paired closing tag
        for tag in img_tags:
            closing_tag = tag.replace('<', '</')
            if closing_tag not in text:
                text = re.sub(r'\s*' + re.escape(tag), '', text)
        text = text.replace(",,", "")
        return text

    @staticmethod
    def clean_tag(s):
        return re.sub('</img\d*>', '', re.sub('<img\d*>', '', s)).strip()


class StringUtils:
    @staticmethod
    def get_digit_num(n):
        return math.ceil(math.log10(n + 1))
    
    @staticmethod
    def format_number(n, dn=None):
        if dn is None:
            dn = StringUtils.get_digit_num(n) + 3
        return "{:0{}}".format(n, dn)
    
    @staticmethod
    def camel_to_snake(name):
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    
    @staticmethod
    def find_all_indices(text, substring):
        indices = []
        index = -1
        while True:
            try:
                index = text.find(substring, index + 1)
            except ValueError:
                pass
            if index == -1:
                break
            indices.append(index)
        return indices


class MPUtils:
    @staticmethod
    def prepare_shards(data, nproc):
        if (len(data) % nproc) == 0:
            ss = len(data) // nproc
        else:
            ss = len(data) // nproc + 1
        
        shards = [data[i*ss:(i+1)*ss] for i in range(nproc)]
        return shards


    @staticmethod
    def mp_func(target_func, args_list):
        nporc = len(args_list)
        processes = []
        for i in range(nporc):
            proc = mp.Process(target=target_func, args=args_list[i])
            proc.start()
            processes.append(proc)
            logging.info("Start process {}".format(i))

        logging.info("Waiting for the finish of all processes")
        for proc in processes:
            proc.join()


class FileUtils:
    @staticmethod
    def exists(file_path):
        return path.exists(file_path)
    
    @staticmethod
    def rename(source_fname, target_fname):
        if os.path.exists(source_fname):
            # Rename the file
            os.rename(source_fname, target_fname)
            logging.info("File renamed from {} to {}".format(source_fname, target_fname))
        else:
            logging("The file {} does not exist".format(source_fname))

    @staticmethod
    def is_dir(file_path):
        return path.isdir(file_path)

    @staticmethod
    def get_last_path(path):
        if not path:
            return path
        parent, last_path = os.path.split(path)
        if last_path:
            return last_path
        else:
            return FileUtils.get_last_path(parent)

    @staticmethod
    def get_dir(path):
        return os.path.dirname(path)

    @staticmethod
    def check_dirs(dir_path):
        if path.exists(dir_path):
            logging.info("{} already exists".format(dir_path))
        else:
            logging.info("Making new directory {}".format(dir_path))
            os.makedirs(dir_path)

    @staticmethod
    def check_basename(fpath):
        bname = os.path.basename(fpath)
        parts = bname.split(".")
        if len(parts) <= 1:
            return bname
        elif parts[-1] in LANGUAGE_LIST or parts[-1] in FileType.ALL:
            return ".".join(parts[:-1])
        else:
            return bname

    @staticmethod
    def check_file_type(fpath):
        parts = fpath.split(".")
        ext = ""
        if parts:
            ext = parts[-1]
        return ext  

    @staticmethod
    def data_iterator(file_pattern, file_type=None, shard_size=0):
        fpath_list = sorted(list(glob(file_pattern)))
        if fpath_list:
            logging.info("Files will be loaded in the following order:\n{}".format(
                "\n".join(fpath_list)
            ))
        else:
            logging.warning("No file found given this pattern: {}".format(file_pattern))
        shard_data = []
        for fpath in fpath_list:
            logging.info("Start to process {}".format(fpath))
            loaded_data = FileUtils.load_file(fpath, file_type)
            if shard_size > 0:
                shard_data += loaded_data
                while len(shard_data) >= shard_size:
                    yield shard_data[:shard_size]
                    shard_data = shard_data[shard_size:]
            else:
                for d in loaded_data:
                    yield d
        if shard_size > 0 and shard_data:
            yield shard_data
        
    @staticmethod
    def load_from_disk(fpath, file_tyle=None):
        return FileUtils.load_file(fpath, file_tyle)

    @staticmethod
    def load_file(fpath, file_type=None):
        if file_type is None:
            file_type = FileUtils.check_file_type(fpath)
        if file_type == FileType.TXT:
            data = []
            with open(fpath, 'r') as fin:
                for line in fin:
                    data.append(line.strip())
        elif file_type == FileType.PT:
            data = torch.load(fpath)
        elif file_type == FileType.JSON:
            with open(fpath, 'r') as fin:
                data = json.load(fin)
        elif file_type == FileType.TSV:
            data = []
            with open(fpath, 'r') as fin:
                for line in fin:
                    data.append(line.strip().split("\t"))
        elif file_type == FileType.CSV:
            data = dict()
            with open(fpath) as fin:
                reader = csv.DictReader(fin)
                col_names = reader.fieldnames
                data = [col_names]
                for row in reader:
                    data.append([row[cn] for cn in col_names])
        elif file_type == FileType.TAR:
            from PIL import Image
            # NOTE: this is only for the caption & img data
            data = []
            with tarfile.open(fpath, "r") as tar:
                txt_files = sorted([f.name for f in tar.getmembers() if f.name.endswith('.txt')])
                jpg_files = [FileUtils.handle_file_extension(it, 'jpg', 'change', True) for it in txt_files]
                for tf, jf in zip(txt_files, jpg_files):
                    txt_obj = tar.extractfile(tf)
                    jpg_obj = tar.extractfile(jf)
                    txt_data = txt_obj.read().decode("utf-8")
                    jpg_data = Image.open(BytesIO(jpg_obj.read()))
                    data.append((txt_data, jpg_data))
        elif file_type == FileType.YAML:
            with open(fpath, 'r') as file:
                data = yaml.load(file, Loader=yaml.FullLoader)
        else:
            logging.warning("Unknown loading file type: {}".format(file_type))
            if file_type in LANGUAGE_LIST:
                data = []
                logging.info("Treat file with language suffix {} by txt".format(file_type))
                with open(fpath, 'r') as fin:
                    for line in fin:
                        data.append(line.strip())
            else:
                data = torch.load(fpath)
        logging.info("Loaded file from {}".format(fpath))
        return data

    @staticmethod
    def save_file(data, fpath, file_type=None):
        FileUtils.save_to_disk(data, fpath, file_type)

    @staticmethod
    def save_to_disk(data, fpath, file_type=None):
        if file_type is None:
            file_type = FileUtils.check_file_type(fpath)

        if file_type == FileType.TXT:
            with open(fpath, 'w') as fout:
                for line in data:
                    fout.write("{}\n".format(line.strip()))
        elif file_type == FileType.PT:
            torch.save(data, fpath)
        elif file_type == FileType.JSON:
            with open(fpath, 'w') as fout:
                json.dump(data, fout, indent="\t")
        elif file_type == FileType.TSV:
            with open(fpath, 'w') as fout:
                for it in data:
                    fout.write("{}\n".format("\t".join(it).strip()))
        elif file_type == FileType.YAML:
            with open(fpath, 'w') as fout:
                yaml.dump(data, fout)
        elif file_type == FileType.CSV:
            with open(fpath, 'w') as fout:
                writer = csv.DictWriter(fout)
                for row in data:
                    writer.writerow(row)
        else:
            logging.warning("Unknown saving file type: {}".format(file_type))
            if file_type in LANGUAGE_LIST:
                logging.info("Treat file with language suffix {} by txt".format(file_type))
                with open(fpath, 'w') as fout:
                    for line in data:
                        fout.write("{}\n".format(line.strip()))
            else:
                torch.save(data, fpath)
        logging.info("Save file to {}".format(fpath))
    
    @staticmethod
    def handle_file_extension(file_path, new_extension, type=FileExtensionType.ADD, only_return_basename=False):
        from pathlib import Path
        # Ensure the new extension starts with a dot
        if not new_extension.startswith("."):
            new_extension = f".{new_extension}"
        file = Path(file_path)
        if type == FileExtensionType.CHANGE:
            new_file_name = f"{file.parent}/{file.stem}{new_extension}"
        elif type == FileExtensionType.ADD:
            new_file_name = f"{file.parent}/{file.stem}{new_extension}{file.suffix}"
        if only_return_basename:
            return os.path.basename(new_file_name)
        else:
            return new_file_name