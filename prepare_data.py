# Copyright (c) 2020 Sarthak Mittal
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import glob
import os
import logging
import traceback
import magic
import pdf2image
from PIL import Image
import simplejson
import tqdm
import multiprocessing as mp

from invoicenet import FIELDS, FIELD_TYPES
from invoicenet.common import util


logger = logging.getLogger(__name__)


def process_file(filename, out_dir, phase, desired_img_w=128, desired_img_h=128):
    try:
        filetype = magic.from_file(filename, mime=True)
        if filetype == "application/pdf":
            page = pdf2image.convert_from_path(filename)[0]
        elif filetype in ("image/jpeg" or "image/jpg" or "image/png"):
            page = Image.open(filename)
        else:
            raise Exception(f"Can't process file, unrecognized file type '{filetype}'")

        output_h_perc = page.size[1] / desired_img_h
        output_w_perc = page.size[0] / desired_img_h
        # print(f"resizing, orig_size = {page.size[1]} desired size = {desired_img_h} perc = {output_h_perc}")
        # print(f"resizing, orig_size = {page.size[0]} desired size = {desired_img_w} perc = {output_w_perc}")
        ngrams = util.create_ngrams(page, output_h_perc, output_w_perc)
        for ngram in ngrams:
            if "amount" in ngram["parses"]:
                ngram["parses"]["amount"] = util.normalize(ngram["parses"]["amount"], key="amount")
            if "date" in ngram["parses"]:
                ngram["parses"]["date"] = util.normalize(ngram["parses"]["date"], key="date")

        page.resize((desired_img_w, desired_img_h))
        page.save(os.path.join(out_dir, phase, os.path.basename(filename)[:-3] + 'jpg'))

        with open(filename[:-3] + 'json', 'r') as fp:
            labels = simplejson.loads(fp.read())

        fields = {}
        for field in FIELDS:
            if field in labels:
                if FIELDS[field] == FIELD_TYPES["amount"]:
                    fields[field] = util.normalize(labels[field], key="amount")
                elif FIELDS[field] == FIELD_TYPES["date"]:
                    fields[field] = util.normalize(labels[field], key="date")
                else:
                    fields[field] = labels[field]
            else:
                fields[field] = ''
        file_name_no_extension = ".".join(os.path.basename(filename).split(".")[:-1])
        data = {
            "fields": fields,
            "nGrams": ngrams,
            "height": desired_img_h,
            "width": desired_img_w,
            "filename": os.path.abspath(
                os.path.join(out_dir, phase, file_name_no_extension + '.jpg'))
        }
        with open(os.path.join(out_dir, phase, file_name_no_extension + '.json'), 'w') as fp:
            fp.write(simplejson.dumps(data, indent=2))
        return True

    except Exception as exp:
        print("Skipping {} : {}".format(filename, exp))
        traceback.print_exc()
        return False


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_dir", type=str, required=True,
                    help="path to directory containing invoice document images")
    ap.add_argument("--out_dir", type=str, default='processed_data/',
                    help="path to save prepared data")
    ap.add_argument("--val_size", type=float, default=0.2,
                    help="validation split ration")
    ap.add_argument("--cores", type=int, help='Number of virtual cores to parallelize over',
                    default=max(1, (mp.cpu_count() - 2) // 2))  # To prevent IPC issues
    ap.add_argument("--ocr_engine", type=str, default='pytesseract',
                    help='OCR used to extract text', choices=['pytesseract', 'aws_textract'])

    args = ap.parse_args()

    os.makedirs(os.path.join(args.out_dir, 'test'), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'val'), exist_ok=True)

    filenames = [os.path.abspath(f) for e in (".pdf", ".png", ".jpeg", ".jpg") for f in glob.glob(args.data_dir + f"**/*{e}", recursive=True) ]
    test_files_idx = int(len(filenames) * 0.05)
    test_files = filenames[:test_files_idx]
    filenames = filenames[test_files_idx:]
    val_files_idx = int(len(filenames) * args.val_size)
    train_files = filenames[val_files_idx:]
    val_files = filenames[:val_files_idx]
    print("Total: {}".format(len(filenames)))
    print("Training: {}".format(len(train_files)))
    print("Validation: {}".format(len(val_files)))

    for phase, filenames in [('train', train_files), ('val', val_files), ('test', test_files)]:
        print("Preparing {} data...".format(phase))

        with tqdm.tqdm(total=len(filenames)) as pbar:
            pool = mp.Pool(args.cores)
            for filename in filenames:
                pool.apply_async(process_file, args=(filename, args.out_dir, phase, 128, 128),
                                 callback=lambda _: pbar.update())

            pool.close()
            pool.join()


if __name__ == '__main__':
    main()
