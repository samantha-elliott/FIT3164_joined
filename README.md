# FIT3164_joined

### Clone the repository
To run Flask and this application locally, first start by cloning this repository onto your local machine.

### Set up a virtual environment
Once you have cloned this repository, please set up a virtual environment inside the folder for this repository. If you do not have a virtual environment installed, please follow the below commands in Terminal (Mac) or cmd (Windows): <br>

`pip install virtualenv` <br>
`python -m virtualenv NAME_OF_YOUR_ENVIRONMENT`

To activate the virtual environment, run <br>
Mac: <br>
`NAME_OF_YOUR_ENVIRONMENT\scripts\activate.bat`<br> 
Windows: <br>
`source NAME_OF_YOUR_ENVIRONMENT/bin/activate` <br>

### Setting up Flask
Set up Flask first by running 
Mac: <br>
`export FLASK_APP=app`<br> 
Windows: <br>
`set FLASK_APP=app` <br>

### Running Flask
To run Flask, run the command <br>
`flask run` <br>
This will run a local instance of the application in the command line. To open the application, copy the url (usually http://127.0.0.1:5000/) and direct it to which page you wish to visit (i.e http://127.0.0.1:5000/home.html)
