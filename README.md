# ChatGPT-Chinese
Unofficial implementation of ChatGPT with minimize code

#### 1. Create wiki dataset:

```bash
export WIKI_ROOT=/data # Use absolute path
mkdir -p ${WIKI_ROOT}
rm -rf ${WIKI_ROOT}/wikicorpus_*
python wikicleaner/wiki_downloader.py --language=zh --save_path=${WIKI_ROOT}
# Or download by chrome then execute: bzip2 -dk ${bz2file}
python wikicleaner/WikiExtractor.py ${WIKI_ROOT}/wikicorpus_zh/wikicorpus_zh.xml -o ${WIKI_ROOT}/wikicorpus_zh/text
cd wikicleaner
bash run.sh "${WIKI_ROOT}/text/*/wiki_??" ${WIKI_ROOT}/wikicorpus_zh/results
cd ..
```
