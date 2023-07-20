#!/bin/sh
declare -a fils=(mediaeval/splits/train_0.txt mediaeval/splits/val.txt mediaeval/splits/test_0.txt clef_en/splits/train_0.txt clef_en/splits/val.txt clef_en/splits/test_0.txt lesa/splits/train_0.txt lesa/splits/val.txt lesa/splits/test_0.txt)

echo "id,label" >> output/combined_labels.csv

for f in "${fils[@]}"
do
	cat "${f}" >> output/combined_labels.csv; echo
done
