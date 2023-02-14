# Adversarial attack against Label propagation

**Setup**
Tested under python `3.6.8` and `3.7.2`.\
Please exctract the svm_light MNIST dataset version archives (can be found in `data` folder)

**Environment install:**\
- `python3 -m venv venv/`
- `source venv/bin/activate`
- `pip install -r requirements.txt`

**Run**\
Define experiement parameters on the `config.yaml` file. (applicable to research question 2,3,4 as RQ1 has static config)

Run the experiment using the corresponding `main_rq*.py` script.
The results outputs are as follow :
- `main_rq1.py` : single file per dataset : `transductive_accuracy_exhaustive_lp_ls_${dataset}_${p_labelled}` representing the accuracy decrease (relative decrease) by flipping each of the labeled samples.
  
- `main_rq2.py` : files : `{inductive_algo}_{ssl_name}_{dataset}_{p_labelled}` representing the absolute accuracy for inducive algo = [rfc,mlp] ssl = [lp,ls] dataset = [mnist,cifar,rcv1] for three different unlabelled proportions p_labelled = [5%,15%,25%] given 5 flip budget = [0%,5%,10%,15%,20%]
- `main_rq3` : same as rq but with attack algorithm = [random,greedy,probabilistic] with time efficiency report.

Plots helpers functions can be found in `representation/main.py`

**Non exhaustive Requirements**\
cycler==0.10.0
decorator==4.4.2
imageio==2.9.0
joblib==0.16.0
kiwisolver==1.2.0
matplotlib==3.3.0
networkx==2.4
numpy==1.19.1
Pillow==7.2.0
pyparsing==2.4.7
python-dateutil==2.8.1
PyWavelets==1.1.1
scikit-image==0.17.2
scikit-learn==0.23.2
scipy==1.5.2
six==1.15.0
threadpoolctl==2.1.0
tifffile==2020.7.24
