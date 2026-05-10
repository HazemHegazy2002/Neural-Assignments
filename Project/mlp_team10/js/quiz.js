const questions=[
  {q:"What does MLP stand for?",opts:["Multi-Layer Perceptron","Multiple Learning Parameters","Machine Learning Pipeline","Multi-Level Processing"],ans:0},
  {q:"What is the minimum number of layers an MLP must have?",opts:["1 layer","2 layers","3 layers","4 layers"],ans:2},
  {q:"Why do we apply an activation function in an MLP?",opts:["To speed up computation","To add nonlinearity so the network can learn complex patterns","To reduce the number of weights","To normalize the input data"],ans:1},
  {q:"What problem can an MLP solve that a single perceptron cannot?",opts:["Linear classification","AND gate","XOR gate","OR gate"],ans:2},
  {q:"What does the bias term (b) do in a neuron?",opts:["It multiplies the input","It shifts the activation function, allowing the neuron to activate even when inputs are zero","It reduces overfitting","It controls the learning rate"],ans:1},
  {q:"Which of the following is NOT an activation function?",opts:["ReLU","Sigmoid","Softmax","Gradient"],ans:3},
  {q:"What is backpropagation used for?",opts:["Computing the forward pass output","Generating training data","Computing gradients to update weights","Initializing the network"],ans:2},
  {q:"What happens if the learning rate is too large?",opts:["The network learns too slowly","The network may overshoot and fail to converge","The network ignores the bias","The loss becomes zero immediately"],ans:1},
  {q:"What is one epoch in neural network training?",opts:["One weight update","One forward pass on a single sample","One complete pass through the entire training dataset","One layer of computation"],ans:2},
  {q:"What is overfitting in an MLP?",opts:["When the model performs well on training data but poorly on new data","When the model has too few neurons","When the learning rate is too small","When the activation function is ReLU"],ans:0},
  {q:"Which activation function outputs values in the range (0, 1)?",opts:["ReLU","Tanh","Sigmoid","Leaky ReLU"],ans:2},
  {q:"What does the loss function measure?",opts:["The number of layers in the network","The speed of training","How far the network's predictions are from the true labels","The learning rate"],ans:2},
  {q:"For image classification, which architecture is generally preferred over MLP?",opts:["RNN","CNN","Decision Tree","Logistic Regression"],ans:1},
  {q:"What is gradient descent?",opts:["A technique to initialize weights randomly","An optimization algorithm that updates weights in the direction that reduces loss","A method to normalize inputs","A way to add more layers"],ans:1},
  {q:"Which technique helps prevent overfitting in an MLP?",opts:["Using more neurons in every layer","Increasing the learning rate","Dropout — randomly disabling neurons during training","Removing the bias term"],ans:2},
];

function buildQuiz(){
  let c=document.getElementById('quiz-container');
  c.innerHTML='';
  questions.forEach((q,i)=>{
    let div=document.createElement('div');
    div.className='quiz-q';
    div.id='q'+i;
    let optsHTML=q.opts.map((o,j)=>`
      <label class="quiz-option" id="opt${i}_${j}">
        <input type="radio" name="q${i}" value="${j}">
        <span>${o}</span>
      </label>`).join('');
    div.innerHTML=`<div class="quiz-q-num">Question ${i+1} of ${questions.length}</div>
      <div class="quiz-q-text">${q.q}</div>
      <div class="quiz-options">${optsHTML}</div>`;
    c.appendChild(div);
  });
}

function submitQuiz(){
  let score=0;
  questions.forEach((q,i)=>{
    let sel=document.querySelector(`input[name="q${i}"]:checked`);
    let chosen=sel?parseInt(sel.value):-1;
    q.opts.forEach((_,j)=>{
      let lbl=document.getElementById(`opt${i}_${j}`);
      if(lbl){
        if(j===q.ans)lbl.classList.add('correct');
        else if(j===chosen)lbl.classList.add('wrong');
        let inp=lbl.querySelector('input');
        if(inp)inp.disabled=true;
      }
    });
    if(chosen===q.ans)score++;
  });
  let pct=Math.round(score/questions.length*100);
  let msg=pct===100?'Perfect score! Excellent understanding of MLP.':pct>=80?'Great job! You have a solid grasp of MLP.':pct>=60?'Good effort. Review the sections you missed.':'Keep studying — re-read the sections above and try again.';
  let res=document.getElementById('quiz-result');
  res.style.display='block';
  res.innerHTML=`<div class="quiz-score">${score}/${questions.length}</div><div class="quiz-msg">${pct}% — ${msg}</div>`;
  document.getElementById('quiz-submit').style.display='none';
  document.getElementById('quiz-reset').style.display='inline-block';
  res.scrollIntoView({behavior:'smooth',block:'center'});
}

function resetQuiz(){
  document.querySelectorAll('.quiz-option').forEach(l=>{l.classList.remove('correct','wrong')});
  document.querySelectorAll('.quiz-options input').forEach(i=>{i.checked=false;i.disabled=false});
  document.getElementById('quiz-result').style.display='none';
  document.getElementById('quiz-submit').style.display='inline-block';
  document.getElementById('quiz-reset').style.display='none';
  document.getElementById('quiz-container').scrollIntoView({behavior:'smooth'});
}

buildQuiz();
