cd /home/liaoyuanjun/cgt_realscene/final
find . -type f ! -name MANIFEST.sha256 | sort | xargs sha256sum > MANIFEST.sha256
sha256sum -c MANIFEST.sha256 --quiet && echo ALL_OK
wc -l MANIFEST.sha256