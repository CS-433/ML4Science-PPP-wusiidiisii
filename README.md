# ML Project: Probe Position Prediction

### Motivation and design

Instead of using the traditional way, namely, the simple stereo or horizontal stereo, to recover the 3D structure of a scene from two 2D images, we perfer to obtain the 3D object position directly from 2D images with Machine Learning methods. <br>
The project is divided into two stages. 

1. Generate ourselves training dataset with the python library *Kubric*. 
2. Train ResNet model according to the dataset we generated in the first step.
3. Visualize the output of our model.

### Github Folder Instruction
1. src: All developed codes
2. probe_models: Probe 3D Construction Data
3. dataset: Generated dataset <br>
4. dataset_split: dataset which has been split into training, validation and test set <br>
5. model_output: output from the NN model <br>
6. prediction: Prediction files according to the model output and the rendering 2D images <br>
7. compare_rgba: comparison of image pairs between the true images and the predict ones


### Getting started
The whole project is developed in Linux operating system. With Windows platform, it is necessary to install *wsl* and use *Ubuntu on Windows* to run the program. The instruction of installation only contains instructions for Linux OS.

##### Install docker
1. Set up the repository<br>

        sudo apt-get update
        sudo apt-get install \
            ca-certificates \
            curl \
            gnupg \
            lsb-release
		sudo mkdir -p /etc/apt/keyrings
		curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
		echo \
  			"deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  			$(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

2. Install Docker Engine<br>

		sudo apt-get update
		sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin

3. Validation the installation

		sudo docker run hello-world

##### Install Kubric Docker
1. Install Kubirc<br>
		
		git clone https://github.com/google-research/kubric.git
		cd kubric

2. Deploy Docker<br>
		
		docker pull kubricdockerhub/kubruntu

3. Validation the installation<br>
		
		docker run --rm --interactive \
           	--user $(id -u):$(id -g) \
           	--volume "$(pwd):/kubric" \
           	kubricdockerhub/kubruntu \
           	/usr/bin/python3 examples/helloworld.py
		ls output

### Generate Dataset
After installed all necessary enviroment, put the folder *ML_Project* and *Models* into the current working path. *ML_Project* contains all developed codes and output. *Models* contains the probe construction data which will be used in dataset generation part. <br>

All used resources in the generation procedure is stored in the Google Store. <br>
<ol>
<li> kubasic assets: gs://kubric-public/assets/KuBasic/
<li> HDRI assets: gs://kubric-public/assets/HDRI_haven/
</ol>
Download the resources to local path or directly use the automatically setting in our code.

The code for generating is located in folder *ML_Project* with file name *Dataset_Generation.py*.
The generation code is:

	cd kubric
	docker run --rm --interactive \            
		--user $(id -u):$(id -g) \            
		--volume "$(pwd):/kubric" \            
		kubricdockerhub/kubruntu \            
		/usr/bin/python3 src/Dataset_Generation.py

It is also possible to set parameters in dataset generation process. <br>

	--r_interval: value of distance change between different r
	--r_change_number: change time of parameter r
	--phi_change_number: change time of parameter phi
	--theta_change_number: change time of parameter theta
	--bg_change_number: change time of background
	--backgrounds_split: select to use backgrounds in training or test sets

### Train Model
1. Split the generated dataset into training, validation and test set

		python3 ./src/merge_dataset.py

2. Training the model on Google Colab

	Put the file "src/run.ipynb" on Google Colab and follow steps in it.


### Visualize the Predicted Position of Probe
1. Generate the prediction file from the model output

		docker run --rm --interactive \           
			--user $(id -u):$(id -g) \            
			--volume "$(pwd):/kubric" \            
			kubricdockerhub/kubruntu \            
			/usr/bin/python3 src/Generate_prediction_file.py

2. Generate the 2D images from the prediction files above

		docker run --rm --interactive \            
			--user $(id -u):$(id -g) \            
			--volume "$(pwd):/kubric" \            
			kubricdockerhub/kubruntu \            
			/usr/bin/python3 src/Generate_2D_Prediction.py

3. Generate the comparison image pairs between the true images and the predicted images

		python3 ./src/compare_rgba.py

### Team Member
- Xingchen Li<br>
- JackRuihang<br>
- shcSteven