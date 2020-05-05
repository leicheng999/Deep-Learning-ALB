for j in $(seq 3)
do

echo 'train_4_10_50_5_50_G0'>> result$j$j.txt
for i in $(seq 25)
do
echo $i >> result$j$j.txt
cat ./alpha$i/result$j.txt |grep -B 1 100% >> result$j$j.txt
echo '\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\' >> result$j$j.txt

done

done
