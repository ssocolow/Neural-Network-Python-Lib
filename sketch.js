let NUMBER_OF_WEIGHTS = 2;
let NUMBER_OF_POINTS = 1000;
let LEARNING_RATE = 0.01;

//Trying to save data
let json = {};
let writer;

let correct = 0;

let points = [];

let b;

let training = false;


function setup() {
  createCanvas(400, 400);
  writer = createWriter('Perceptron_data.txt');

  //Making the points
  for (let i = 0; i < NUMBER_OF_POINTS; i++){
  	points[i] = new Point(); 
	}
  
  //Making the Perceptron
  b = new Perceptron();
  
  //Adding initial weights to JSON
  json.weights = b.weights;
  json.learning_rate = b.learning_rate;
}

function draw() {
  correct = 0;
  
  background(255);
  stroke(0);
  
  line(0,height,width,0);
  
  //Show points
  for(let point of points){
    point.show();
  }
  
  //Train the network for each of the points
  for(let pt of points){
    let inputs = [pt.x/width, pt.y/height];
    let target = pt.label;
   
    //Train it when mouse is clicked
    b.train(inputs, target)
    if(training){
  		b.train(inputs, target);
		}
  
    let guess = b.guess(inputs);
    if(guess == target) {
      fill(0,255,0);
      correct++;
    } else {
      fill(255,0,0);
    }
    noStroke();
    ellipse(pt.pixelX(),pt.pixelY(),4,4);
  }
  	//Stop it from training until the next mouse click
    training = false;
  	document.getElementById("error").innerHTML = correct;
  	document.getElementById("weights").innerHTML = b.weights;

  
}

function mousePressed(){
  //Train the Perceptron
  training = true;
  //Prints the percent correctness before the next training cycle
  print(correct/NUMBER_OF_POINTS *100 + '%');
  print(b);
}
