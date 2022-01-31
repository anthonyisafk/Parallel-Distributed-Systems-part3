for size in 16 64 256 512 1024;
do 
  for i in {1..10};
  do
  ./v0.out $size 1000;
  done;
done