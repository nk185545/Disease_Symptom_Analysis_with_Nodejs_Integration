# Disease_Symptom_Analysis_with_Nodejs_Integration

Application deployed link :  https://disease-symptom-analysis-nk.herokuapp.com/


#### 1) Machine Learning Part :--

step 1 ) First of all data scrapping is done using **GoogleSearch and BeautifulSoup libraries** from the url : https://www.nhp.gov.in/disease-a-z/ . Stored the disease and symptoms.

step 2 ) Then cleaned this data and make combinations and finally build the csv file.

step 3 ) Applied Convolutional Neural Network (CNN) to train and test the model using  **Tensorflow**.

step 4 ) Saved the model using **Tensorflow.js** library.

**--------------------------------------------------------------------**

#### 2) Node.js part :--

Built a normal Node.js project and integrate the ML Model with tensorflow.js library.

//  Load libraries 

const tf = require("@tensorflow/tfjs") 

const tfn = require("@tensorflow/tfjs-node");


//   Load tensorflowjs model

var model ;

const handler = tfn.io.fileSystem(tfjsModelPath);   //  tfjsModelPath -  path to tensorflow.js model (model.json)

(async function(){
    model = await tf.loadLayersModel(handler);   
})() ;


//    Prediction

let predictions = await model.predict(tf.tensor3d(input)).data();



In this application, minimum 2 symptoms or maximum 4 symptoms can be entered then our application will clean the data and make the data in the form which is suitable to predict function and finally  display the three disease with top 3 probabilities. 


**--------------------------------------------------------------------------------------------------**

**Application deployed link** :  https://disease-symptom-analysis-nk.herokuapp.com/

**--------------------------------------------------------------------------------------------------**
