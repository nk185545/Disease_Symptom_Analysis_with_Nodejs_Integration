const express = require("express") ;
const path = require("path") ;
const hbs = require("hbs") ;
const app = express() ;
const port = process.env.PORT || 8000 ;

const bodyParser = require('body-parser');

const tf = require("@tensorflow/tfjs")
const tfn = require("@tensorflow/tfjs-node");
const natural = require('natural');
const lem = require("lemmatizer")


const {stop_words} = require("../tfjs/stopWords")
const {disease_arr} = require("../tfjs/disease_list")
const {symptom_arr} = require("../tfjs/symptom_list")


app.use(bodyParser.json());
app.use(bodyParser.urlencoded({
  extended: true
}));


//public static path
const staticPath = path.join(__dirname, "../public") ;
const template_path = path.join(__dirname, "../templates/views") ;
const partials_Path = path.join(__dirname, "../templates/partials") ;
const tfjsModelPath = path.join(__dirname, "../tfjs/dsa_tfjs_model/model.json") ;

/*---load tensorflowjs model-----*/
var model ;
const handler = tfn.io.fileSystem(tfjsModelPath);

(async function(){
    model = await tf.loadLayersModel(handler);    // $('.progress-bar').hide() ; 
})() ;

app.set("view engine","hbs") ;
app.set("views",template_path)

// hbs register partials
hbs.registerPartials(partials_Path) ;

app.use(express.static(staticPath)) ;


// console.log(tf.tensor1d([1, 2, 3]));


app.get("",(req,res)=> {
    res.render('index')
}) ;

app.post("/predict_disease",async function(req,res){
    let ip1 = req.body.input1
    let ip2 = req.body.input2
    let ip3 = req.body.input3
    let ip4 = req.body.input4    

    let symptoms = ip1+", "+ip2+", "+ip3+", "+ip4;
    symptoms = symptoms.toLowerCase()

    //symptoms = symptoms.replace(/[^A-Za-z0-9]/g," ")

    let symp_list = symptoms.split(",");
    let temp_list=[]
    for(let sym of symp_list){
        if(sym.trim().length >0){
            temp_list.push(sym.trim())
        }
    }
    if("none" in temp_list) 
    {
        temp_list.remove("none")
    }
    
    let all_symp = []
    for(let sym of temp_list){

        sym=sym.replace('-',' ')
        sym=sym.replace("'",'')
        sym=sym.replace('(','')
        sym=sym.replace(')','')
        sym = sym.replace(/[^A-Za-z]/g," ")  // remove all non-alphanumeric characters

        let wordlist = sym.split(" ")
        
        let cleaned_sym="" ;
        for(let wd of wordlist){
            if(stop_words.indexOf(wd)==-1){
                let temp = lem.lemmatizer(wd) ;
                cleaned_sym = cleaned_sym+wd+" ";
            }
        }
        cleaned_sym = cleaned_sym.trim() ;
        all_symp.push(cleaned_sym) ;
        
    }
    
    let data_arr=[];
    let one=[]
    one.push(1)
    let zero=[]
    zero.push(0)
    for(let single_sym of symptom_arr){
        if(all_symp.indexOf(single_sym)!==-1){
            data_arr.push(one) ;
        }
        else{
            data_arr.push(zero) ;
        }
    }

    let finaldata = []
    finaldata.push(data_arr)
    
    let predictions = await model.predict(tf.tensor3d(finaldata)).data();
    //console.log(predictions)
    
    let top3 = Array.from(predictions)
        .map(function(p,i){
            return {
                probability:p ,
                className:disease_arr[i]
            };
        }).sort(function(a,b){
            return b.probability - a.probability
        }).slice(0, 3);


        top3.forEach(function(p){
            console.log(`${p.className}  :  ${p.probability.toFixed(6)} `)
        });
    //console.log(top3)

    res.render('index',{top3:top3})
}) ;



app.listen(port,() => {
    console.log(`listening to the port at ${port} `)
})

