if [[ $# -ne 4 ]]
then
echo "Not enough arguments present!!"
exit 1
fi
QN=$1
MN=$2
IF=$3
OF=$4
if [ -e OF ]
then
	rm $4
fi
if [ $QN -eq 1 ]
then
	if [ $MN -eq 1 ]
	then 
		python3 nb_1.py $IF $OF
	fi
	if [ $MN -eq 2 ]
	then
		python3 nb_2.py $IF $OF
	fi
	if [ $MN -eq 3 ]
	then
		python3 nb_3.py $IF $OF
	fi
fi
if [ $QN -eq 2 ]
then
	if [ $MN -eq 1 ]
	then
		python3 svm_1.py $IF $OF
	fi
	if [ $MN -eq 2 ]
	then
		python3 svm_2.py $IF output.txt
		svm-scale -l 0 -u 1 output.txt > scaled_output.scale
		svm-predict scaled_output.scale data/train_lin.scale.model $OF
	fi
	if [ $MN -eq 3 ]
	then
		python3 svm_2.py $IF output.txt
		svm-scale -l 0 -u 1 output.txt > scaled_output.scale
		svm-predict scaled_output.scale data/train_best.scale.model $OF
	fi
fi