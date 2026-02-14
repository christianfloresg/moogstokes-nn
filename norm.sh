for fname in data/science/*.fits; do
	python specnorm_proplyd.py "$fname"
done
