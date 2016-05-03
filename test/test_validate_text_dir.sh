#!/bin/bash


rm -rf foo
mkdir -p foo
echo "hi now then" > foo/dev.txt
echo "now and then" >> foo/dev.txt

# the following should fail
echo "Ignore the following error:"
../scripts/validate_text_dir.py foo && exit 1;

echo "Ignore the following error:"
echo -n > foo/train1.txt
../scripts/validate_text_dir.py foo && exit 1;

echo "Ignore the following error:"
echo '<s>' > foo/train1.txt
../scripts/validate_text_dir.py foo && exit 1;

echo "Ignore the following error:"
echo '</S>' > foo/train1.txt
../scripts/validate_text_dir.py foo && exit 1;

echo "there there" > foo/train1.txt
echo "now now then" >> foo/train1.txt

echo "hi" > foo/train2.txt
echo "hi lo" >> foo/train2.txt

../scripts/validate_text_dir.py foo || exit 1;
rm -r bar
../scripts/get_counts.py foo bar || exit 1

gzip foo/*.txt
../scripts/validate_text_dir.py foo || exit 1;

rm -r bar
../scripts/get_counts.py foo bar || exit 1

../scripts/get_unigram_weights.py --verbose=true bar > weights || exit 1

../scripts/get_vocab.py --weights=weights bar > words.txt

../scripts/validate_vocab.py words.txt || exit 1

(echo hi hello there; echo fdasf; echo '<s>') | ../scripts/text_to_int.py words.txt || exit 1

../scripts/prepare_int_data.sh foo words.txt int || exit 1

echo "Success"


exit 0;
