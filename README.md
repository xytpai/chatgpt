# ChatGPT-Chinese
Unofficial implementation of ChatGPT with minimize code

#### 1. Create wiki dataset:

```bash
export WIKI_ROOT=/data/wiki-chinese
mkdir -p ${WIKI_ROOT}
python wikicleaner/wiki_downloader.py --language=zh --save_path=${WIKI_ROOT}
# Or download by chrome then execute: bzip2 -dk ${bz2file}
python wikicleaner/WikiExtractor.py ${WIKI_ROOT}/wikicorpus_zh.xml -o ${WIKI_ROOT}/text
cd wikicleaner
bash run.sh "${WIKI_ROOT}/text/*/wiki_??" ${WIKI_ROOT}/results
cd ..
```
