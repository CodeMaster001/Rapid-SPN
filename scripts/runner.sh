mkdir results
nohup python iris.py &
cp nohup.out results/iris/iris.txt
wait
nohup python prestige.py  &
cp nohup.out results/prestige/prestige.txt
wait
nohup python hayes_roth.py  &
cp nohup.out results/hayes/hayes.txt
wait
nohup python glass.py  &
cp nohup.out results/glass/glass.txt
wait
nohup python balance_scale.py  &
cp nohup.out results/balance/balance_scale.txt
wait
nohup python ionosphere.py  &
cp nohup.out results/ionosphere/ionosphere.txt
wait
nohup python  cars.py  &
cp nohup.out results/cars/cars.txt
wait
nohup python  farmland.py  &
cp nohup.out results/farmland/farmland.txt
wait

nohup python  pasture.py  &
cp nohup.out results/pasture/pasture.txt


nohup python  autos.py  &
cp nohup.out results/autos/pasture.txt
wait

nohup python  cropland.py  &
cp nohup.out results/cropland/cropland.txt
wait


nohup python  liver_disorder.py  &
cp nohup.out results/liver/liver.txt
wait

nohup python caesarian.py &
cp nohup.out results/caesarian/caesarian.txt

nohup python autos.py &
cp nohup.out results/autos/autos.txt

nohup python BNG_mv.py &
cp nohup.out results/bngmv/bngmv.txt



nohup python thoracic-surgery.py &
cp nohup.out results/surgery/surgery.txt




