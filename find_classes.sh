for i in {0..53}; do
    grep -rnw data/labels/train/ -e "^$i" | head -n 1 | cut -d':' -f1 | xargs -n 1 basename
done
