import argparse

from huggingface_hub import snapshot_download


def main(args):

    snapshot_download(
        repo_id='MBZUAI/GranD-f',
        repo_type='dataset',
        ignore_patterns=[".gitattributes", "README.md", "*.jpg", "*.fp16.*", "*non_ema*"],
        local_dir='/data00/datasets/GranDf',
        local_dir_use_symlinks=False,
        max_workers=4,
        resume_download=True,
        cache_dir='./cache_dir/'
)


if __name__ == '__main__':
    """
    HF_ENDPOINT=https://hf-mirror.com python hf_down.py
    """
    import os
    # os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    parser = argparse.ArgumentParser()
    # parser.add_argument("repo_id")
    # parser.add_argument('local_dir')
    # parser.add_argument('--repo_type', default='model')
    args = parser.parse_args()
    main(args)