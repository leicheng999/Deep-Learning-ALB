for i in $(seq 25)
do cp -r test1 alpha$i
done




for k in $(seq 2 4)
do

for j in $(seq 5)
do

for i in $(seq 5)
do
CUDA_VISIBLE_DEVICES=1 python ./alpha$((i+5*(j-1)))/1234.py --step$k --alpha $((i+5*(j-1))) --l1depth 10 --l1node 50 --l2depth 4 --l2node 50 >> ./alpha$((i+5*(j-1)))/result$k.txt &
done
wait

done

done







