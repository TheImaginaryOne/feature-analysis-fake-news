import os, random
from tqdm import tqdm
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

from multiprocessing import Process
from multiprocessing import Pool, cpu_count

base_dir = 'public_image_set/'


def check_filetype(dest):
    dest = base_dir + dest
    try:
        img = Image.open(dest).load()
    except Exception as e:
        return (e, dest)
    return (None, dest)

def run():
    file_names = os.listdir(base_dir)

    images_list = open("output/_images_list.txt", "w")
    print("n images: " , len(file_names))

    found_ids = set()
    required_fnames = []
    for fn in file_names:
        fn_id = fn.split(".")[0] 
        if fn_id not in found_ids:
            required_fnames.append(fn)
            found_ids.add(fn_id)
    n = len(required_fnames)

    with Pool(20) as p, tqdm(total=n) as pbar:
        procs = [p.apply_async(check_filetype, args=(fn,), callback=lambda _: pbar.update(1)) for fn in file_names] 
        
        for proc in procs:
            status, dest = proc.get()
            if status != None:
                tqdm.write(f"{dest} has an error: {status}")
                continue
            # ../ because the vs analyser is in a subdir.
            images_list.write('../' + dest + '\n')

    

if __name__ == '__main__':
    run()
