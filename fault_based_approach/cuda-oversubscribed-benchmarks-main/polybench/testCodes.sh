# This script takes the same arguments as the binaries

for dir in [0-9A-Z]*
do
	if [ -d $dir ]
	then
		cd $dir
		pwd
		make
		if [ $? -ne 0 ]
		then
			echo ========================================
			echo Trouble over here in compilation: $PWD
			echo ========================================
		fi
		./*exe -compare $@ &
		if [ $? -ne 0 ]
		then
			echo ========================================
			echo Trouble over here in execution: $PWD
			echo ========================================
		fi
		make clean
		cd ..
	fi
done
