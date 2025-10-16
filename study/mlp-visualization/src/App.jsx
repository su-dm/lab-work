import React, { useState, useEffect } from 'react';
import { Play, Pause, RotateCcw, ChevronLeft, ChevronRight } from 'lucide-react';

export default function MLPFeedForwardDetailed() {
  // Network structure: 3 inputs -> 4 hidden -> 2 outputs
  const [animationStep, setAnimationStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [activeNeuron, setActiveNeuron] = useState(null);
  const [highlightedWeights, setHighlightedWeights] = useState([]);
  const [computation, setComputation] = useState(null);
  
  const MAX_STEPS = 16; // 4 hidden neurons × 2 steps + 2 output neurons × 2 steps
  
  // Sample data
  const inputs = [0.5, 0.8, 0.3];
  const inputLabels = ['x₁', 'x₂', 'x₃'];
  
  // Weights: weightsInputHidden[from_input][to_hidden]
  const weightsInputHidden = [
    [0.2, 0.5, -0.3, 0.4],   // from input 0 to each hidden neuron
    [0.1, -0.2, 0.6, 0.3],   // from input 1 to each hidden neuron
    [0.4, 0.3, -0.1, 0.5]    // from input 2 to each hidden neuron
  ];
  const biasHidden = [0.1, -0.1, 0.2, 0.0];
  
  // Weights: weightsHiddenOutput[from_hidden][to_output]
  const weightsHiddenOutput = [
    [0.3, -0.4],   // from hidden 0 to each output
    [0.5, 0.2],    // from hidden 1 to each output
    [-0.2, 0.6],   // from hidden 2 to each output
    [0.4, 0.1]     // from hidden 3 to each output
  ];
  const biasOutput = [0.1, -0.2];
  
  // Sigmoid activation
  const sigmoid = (x) => 1 / (1 + Math.exp(-x));
  
  // Calculate hidden layer values
  const hiddenRaw = biasHidden.map((bias, j) => {
    let sum = bias;
    for (let i = 0; i < inputs.length; i++) {
      sum += inputs[i] * weightsInputHidden[i][j];
    }
    return sum;
  });
  const hidden = hiddenRaw.map(sigmoid);
  
  // Calculate output layer values
  const outputRaw = biasOutput.map((bias, j) => {
    let sum = bias;
    for (let i = 0; i < hidden.length; i++) {
      sum += hidden[i] * weightsHiddenOutput[i][j];
    }
    return sum;
  });
  const output = outputRaw.map(sigmoid);
  
  // Animation control
  useEffect(() => {
    if (isPlaying) {
      const timer = setTimeout(() => {
        setAnimationStep((step) => {
          if (step >= MAX_STEPS) {
            setIsPlaying(false);
            return MAX_STEPS;
          }
          return step + 1;
        });
      }, 800);
      return () => clearTimeout(timer);
    }
  }, [isPlaying, animationStep]);
  
  const handlePlayPause = () => {
    if (animationStep >= MAX_STEPS) {
      setAnimationStep(0);
    }
    setIsPlaying(!isPlaying);
  };
  
  const handleReset = () => {
    setAnimationStep(0);
    setIsPlaying(false);
    setActiveNeuron(null);
    setHighlightedWeights([]);
    setComputation(null);
  };
  
  const handlePrevStep = () => {
    setIsPlaying(false);
    setAnimationStep(Math.max(0, animationStep - 1));
  };
  
  const handleNextStep = () => {
    setIsPlaying(false);
    setAnimationStep(Math.min(MAX_STEPS, animationStep + 1));
  };
  
  // Animation logic
  useEffect(() => {
    const step = animationStep;
    
    if (step > MAX_STEPS) {
      setAnimationStep(MAX_STEPS);
      return;
    }
    
    // Process hidden neurons (4 neurons × 2 steps = 8 steps)
    if (step < 8) {
      const neuronIdx = Math.floor(step / 2);
      const isActivation = step % 2 === 1;
      
      setActiveNeuron({ layer: 'hidden', index: neuronIdx });
      
      if (!isActivation) {
        // Highlight all incoming weights
        const weights = [];
        for (let i = 0; i < inputs.length; i++) {
          weights.push({
            from: 'input',
            fromIdx: i,
            to: 'hidden',
            toIdx: neuronIdx,
            weight: weightsInputHidden[i][neuronIdx]
          });
        }
        setHighlightedWeights(weights);
        
        // Build computation
        let sum = biasHidden[neuronIdx];
        const terms = [`b = ${biasHidden[neuronIdx].toFixed(2)}`];
        for (let i = 0; i < inputs.length; i++) {
          const product = inputs[i] * weightsInputHidden[i][neuronIdx];
          terms.push(`${inputLabels[i]} × w${i+1} = ${inputs[i].toFixed(2)} × ${weightsInputHidden[i][neuronIdx].toFixed(2)} = ${product.toFixed(3)}`);
          sum += product;
        }
        
        setComputation({
          type: 'weighted_sum',
          layer: 'hidden',
          neuron: neuronIdx,
          terms: terms,
          total: sum
        });
      } else {
        // Activation step
        setHighlightedWeights([]);
        const activated = sigmoid(hiddenRaw[neuronIdx]);
        setComputation({
          type: 'activation',
          layer: 'hidden',
          neuron: neuronIdx,
          input: hiddenRaw[neuronIdx],
          output: activated
        });
      }
    }
    // Process output neurons (2 neurons × 2 steps = 4 steps)
    else if (step >= 8 && step < 16) {
      const neuronIdx = Math.floor((step - 8) / 2);
      const isActivation = (step - 8) % 2 === 1;
      
      setActiveNeuron({ layer: 'output', index: neuronIdx });
      
      if (!isActivation) {
        const weights = [];
        for (let i = 0; i < hidden.length; i++) {
          weights.push({
            from: 'hidden',
            fromIdx: i,
            to: 'output',
            toIdx: neuronIdx,
            weight: weightsHiddenOutput[i][neuronIdx]
          });
        }
        setHighlightedWeights(weights);
        
        let sum = biasOutput[neuronIdx];
        const terms = [`b = ${biasOutput[neuronIdx].toFixed(2)}`];
        for (let i = 0; i < hidden.length; i++) {
          const product = hidden[i] * weightsHiddenOutput[i][neuronIdx];
          terms.push(`h${i+1} × w${i+1} = ${hidden[i].toFixed(3)} × ${weightsHiddenOutput[i][neuronIdx].toFixed(2)} = ${product.toFixed(3)}`);
          sum += product;
        }
        
        setComputation({
          type: 'weighted_sum',
          layer: 'output',
          neuron: neuronIdx,
          terms: terms,
          total: sum
        });
      } else {
        setHighlightedWeights([]);
        const activated = sigmoid(outputRaw[neuronIdx]);
        setComputation({
          type: 'activation',
          layer: 'output',
          neuron: neuronIdx,
          input: outputRaw[neuronIdx],
          output: activated
        });
      }
    }
    // Complete state
    else if (step === 16) {
      setActiveNeuron(null);
      setHighlightedWeights([]);
      setComputation({
        type: 'complete',
        outputs: output
      });
    }
  }, [animationStep]);
  
  // Layout
  const layerX = { input: 120, hidden: 450, output: 780 };
  const getNodeY = (idx, total) => 120 + (idx * 450 / (total + 1));
  
  const isWeightHighlighted = (from, fromIdx, to, toIdx) => {
    return highlightedWeights.some(w => 
      w.from === from && w.fromIdx === fromIdx && w.to === to && w.toIdx === toIdx
    );
  };
  
  return (
    <div className="w-full min-h-screen bg-gradient-to-br from-slate-900 to-slate-800 flex flex-col items-center justify-center p-8">
      <h1 className="text-4xl font-bold text-white mb-2">Multi-Layer Perceptron: Feed Forward Algorithm</h1>
      <p className="text-slate-300 mb-6">Detailed view of weights, biases, and computations</p>
      
      <div className="bg-slate-800 rounded-lg shadow-2xl p-8 mb-6">
        <svg width="950" height="600" className="overflow-visible">
          {/* Draw all NON-highlighted connections first (Input -> Hidden) */}
          {inputs.map((_, i) => 
            Array(4).fill(0).map((_, j) => {
              const x1 = layerX.input + 35;
              const y1 = getNodeY(i, 3);
              const x2 = layerX.hidden - 35;
              const y2 = getNodeY(j, 4);
              const midX = (x1 + x2) / 2;
              const midY = (y1 + y2) / 2;
              const weight = weightsInputHidden[i][j];
              const isHighlighted = isWeightHighlighted('input', i, 'hidden', j);
              const isVisible = animationStep > 0;
              
              // Only render if NOT highlighted (highlighted ones will be drawn later)
              if (isHighlighted) return null;
              
              return (
                <g key={`ih-${i}-${j}`}>
                  <line
                    x1={x1}
                    y1={y1}
                    x2={x2}
                    y2={y2}
                    stroke="#475569"
                    strokeWidth="1.5"
                    opacity={isVisible ? 0.4 : 0.15}
                    className="transition-all duration-300"
                  />
                  {isVisible && (
                    <g>
                      <rect
                        x={midX - 22}
                        y={midY - 10}
                        width="44"
                        height="20"
                        fill="#1e293b"
                        stroke="#334155"
                        strokeWidth="1.5"
                        rx="3"
                        opacity={0.7}
                      />
                      <text
                        x={midX}
                        y={midY + 5}
                        fill="#94a3b8"
                        fontSize="11"
                        textAnchor="middle"
                      >
                        {weight.toFixed(2)}
                      </text>
                    </g>
                  )}
                </g>
              );
            })
          )}
          
          {/* Draw all NON-highlighted connections (Hidden -> Output) */}
          {Array(4).fill(0).map((_, i) =>
            Array(2).fill(0).map((_, j) => {
              const x1 = layerX.hidden + 35;
              const y1 = getNodeY(i, 4);
              const x2 = layerX.output - 35;
              const y2 = getNodeY(j, 2);
              const midX = (x1 + x2) / 2;
              const midY = (y1 + y2) / 2;
              const weight = weightsHiddenOutput[i][j];
              const isHighlighted = isWeightHighlighted('hidden', i, 'output', j);
              const isVisible = animationStep >= 8;
              
              // Only render if NOT highlighted
              if (isHighlighted) return null;
              
              return (
                <g key={`ho-${i}-${j}`}>
                  <line
                    x1={x1}
                    y1={y1}
                    x2={x2}
                    y2={y2}
                    stroke="#475569"
                    strokeWidth="1.5"
                    opacity={isVisible ? 0.4 : 0.15}
                    className="transition-all duration-300"
                  />
                  {isVisible && (
                    <g>
                      <rect
                        x={midX - 22}
                        y={midY - 10}
                        width="44"
                        height="20"
                        fill="#1e293b"
                        stroke="#334155"
                        strokeWidth="1.5"
                        rx="3"
                        opacity={0.7}
                      />
                      <text
                        x={midX}
                        y={midY + 5}
                        fill="#94a3b8"
                        fontSize="11"
                        textAnchor="middle"
                      >
                        {weight.toFixed(2)}
                      </text>
                    </g>
                  )}
                </g>
              );
            })
          )}
          
          {/* Input Layer */}
          <text x={layerX.input} y="40" fill="#94a3b8" fontSize="16" fontWeight="bold" textAnchor="middle">Input Layer</text>
          {inputs.map((val, i) => (
            <g key={`input-${i}`}>
              <circle
                cx={layerX.input}
                cy={getNodeY(i, 3)}
                r="30"
                fill="#1e293b"
                stroke="#3b82f6"
                strokeWidth="3"
              />
              <text x={layerX.input} y={getNodeY(i, 3) - 5} fill="#60a5fa" fontSize="12" textAnchor="middle">
                {inputLabels[i]}
              </text>
              <text x={layerX.input} y={getNodeY(i, 3) + 10} fill="#60a5fa" fontSize="18" fontWeight="bold" textAnchor="middle">
                {val.toFixed(1)}
              </text>
            </g>
          ))}
          
          {/* Hidden Layer */}
          <text x={layerX.hidden} y="40" fill="#94a3b8" fontSize="16" fontWeight="bold" textAnchor="middle">Hidden Layer (ReLU/Sigmoid)</text>
          {Array(4).fill(0).map((_, i) => {
            const isActive = activeNeuron?.layer === 'hidden' && activeNeuron?.index === i;
            const showValue = animationStep > i * 2 + 1;
            return (
              <g key={`hidden-${i}`}>
                <circle
                  cx={layerX.hidden}
                  cy={getNodeY(i, 4)}
                  r="30"
                  fill={isActive ? "#1e40af" : "#1e293b"}
                  stroke={isActive ? "#60a5fa" : "#8b5cf6"}
                  strokeWidth={isActive ? "4" : "3"}
                  className="transition-all duration-300"
                />
                {/* Bias label */}
                <text
                  x={layerX.hidden - 50}
                  y={getNodeY(i, 4) + 4}
                  fill="#a78bfa"
                  fontSize="11"
                  fontWeight="bold"
                >
                  b={biasHidden[i].toFixed(2)}
                </text>
                <text x={layerX.hidden} y={getNodeY(i, 4) - 5} fill="#a78bfa" fontSize="12" textAnchor="middle">
                  h{i+1}
                </text>
                {showValue && (
                  <text x={layerX.hidden} y={getNodeY(i, 4) + 10} fill="#a78bfa" fontSize="16" fontWeight="bold" textAnchor="middle">
                    {hidden[i].toFixed(2)}
                  </text>
                )}
              </g>
            );
          })}
          
          {/* Output Layer */}
          <text x={layerX.output} y="40" fill="#94a3b8" fontSize="16" fontWeight="bold" textAnchor="middle">Output Layer</text>
          {Array(2).fill(0).map((_, i) => {
            const isActive = activeNeuron?.layer === 'output' && activeNeuron?.index === i;
            const showValue = animationStep > 8 + i * 2 + 1;
            return (
              <g key={`output-${i}`}>
                <circle
                  cx={layerX.output}
                  cy={getNodeY(i, 2)}
                  r="30"
                  fill={isActive ? "#1e40af" : "#1e293b"}
                  stroke={isActive ? "#60a5fa" : "#10b981"}
                  strokeWidth={isActive ? "4" : "3"}
                  className="transition-all duration-300"
                />
                {/* Bias label */}
                <text
                  x={layerX.output - 50}
                  y={getNodeY(i, 2) + 4}
                  fill="#34d399"
                  fontSize="11"
                  fontWeight="bold"
                >
                  b={biasOutput[i].toFixed(2)}
                </text>
                <text x={layerX.output} y={getNodeY(i, 2) - 5} fill="#34d399" fontSize="12" textAnchor="middle">
                  y{i+1}
                </text>
                {showValue && (
                  <text x={layerX.output} y={getNodeY(i, 2) + 10} fill="#34d399" fontSize="18" fontWeight="bold" textAnchor="middle">
                    {output[i].toFixed(2)}
                  </text>
                )}
              </g>
            );
          })}
          
          {/* NOW draw HIGHLIGHTED connections on top (Input -> Hidden) */}
          {inputs.map((_, i) => 
            Array(4).fill(0).map((_, j) => {
              const x1 = layerX.input + 35;
              const y1 = getNodeY(i, 3);
              const x2 = layerX.hidden - 35;
              const y2 = getNodeY(j, 4);
              const midX = (x1 + x2) / 2;
              const midY = (y1 + y2) / 2;
              const weight = weightsInputHidden[i][j];
              const isHighlighted = isWeightHighlighted('input', i, 'hidden', j);
              
              // Only render if highlighted
              if (!isHighlighted) return null;
              
              return (
                <g key={`ih-highlight-${i}-${j}`}>
                  <line
                    x1={x1}
                    y1={y1}
                    x2={x2}
                    y2={y2}
                    stroke="#60a5fa"
                    strokeWidth="4"
                    opacity={1}
                    className="transition-all duration-300"
                  />
                  <rect
                    x={midX - 22}
                    y={midY - 10}
                    width="44"
                    height="20"
                    fill="#1e40af"
                    stroke="#60a5fa"
                    strokeWidth="2"
                    rx="3"
                  />
                  <text
                    x={midX}
                    y={midY + 5}
                    fill="#60a5fa"
                    fontSize="11"
                    fontWeight="bold"
                    textAnchor="middle"
                  >
                    {weight.toFixed(2)}
                  </text>
                </g>
              );
            })
          )}
          
          {/* NOW draw HIGHLIGHTED connections on top (Hidden -> Output) */}
          {Array(4).fill(0).map((_, i) =>
            Array(2).fill(0).map((_, j) => {
              const x1 = layerX.hidden + 35;
              const y1 = getNodeY(i, 4);
              const x2 = layerX.output - 35;
              const y2 = getNodeY(j, 2);
              const midX = (x1 + x2) / 2;
              const midY = (y1 + y2) / 2;
              const weight = weightsHiddenOutput[i][j];
              const isHighlighted = isWeightHighlighted('hidden', i, 'output', j);
              
              // Only render if highlighted
              if (!isHighlighted) return null;
              
              return (
                <g key={`ho-highlight-${i}-${j}`}>
                  <line
                    x1={x1}
                    y1={y1}
                    x2={x2}
                    y2={y2}
                    stroke="#60a5fa"
                    strokeWidth="4"
                    opacity={1}
                    className="transition-all duration-300"
                  />
                  <rect
                    x={midX - 22}
                    y={midY - 10}
                    width="44"
                    height="20"
                    fill="#1e40af"
                    stroke="#60a5fa"
                    strokeWidth="2"
                    rx="3"
                  />
                  <text
                    x={midX}
                    y={midY + 5}
                    fill="#60a5fa"
                    fontSize="11"
                    fontWeight="bold"
                    textAnchor="middle"
                  >
                    {weight.toFixed(2)}
                  </text>
                </g>
              );
            })
          )}
        </svg>
      </div>
      
      {/* Computation Display */}
      <div className="bg-slate-700 rounded-lg p-6 w-full max-w-5xl min-h-48 text-white mb-6">
        {computation?.type === 'weighted_sum' && (
          <div>
            <h3 className="text-2xl font-bold text-blue-400 mb-4">
              Step {animationStep + 1}: Computing {computation.layer === 'hidden' ? 'Hidden' : 'Output'} Neuron {computation.neuron + 1}
            </h3>
            <div className="bg-slate-800 rounded p-4 mb-3">
              <p className="text-lg font-semibold text-yellow-300 mb-2">Weighted Sum Formula:</p>
              <p className="text-2xl font-mono text-green-400">z = b + Σ(xᵢ × wᵢ)</p>
            </div>
            <div className="bg-slate-800 rounded p-4">
              <p className="text-lg font-semibold text-yellow-300 mb-3">Calculation:</p>
              {computation.terms.map((term, idx) => (
                <p key={idx} className="font-mono text-base text-slate-200 mb-1 ml-4">
                  {idx === 0 ? '• ' : '+ '}{term}
                </p>
              ))}
              <div className="border-t border-slate-600 mt-3 pt-3">
                <p className="font-mono text-xl text-yellow-400">
                  <strong>z = {computation.total.toFixed(4)}</strong>
                </p>
              </div>
            </div>
          </div>
        )}
        
        {computation?.type === 'activation' && (
          <div>
            <h3 className="text-2xl font-bold text-purple-400 mb-4">
              Step {animationStep + 1}: Applying Activation Function
            </h3>
            <div className="bg-slate-800 rounded p-4 mb-3">
              <p className="text-lg font-semibold text-yellow-300 mb-2">Sigmoid Function:</p>
              <p className="text-2xl font-mono text-green-400">σ(z) = 1 / (1 + e^(-z))</p>
            </div>
            <div className="bg-slate-800 rounded p-4">
              <p className="text-lg font-semibold text-yellow-300 mb-3">Calculation:</p>
              <p className="font-mono text-lg text-slate-200 ml-4 mb-2">
                σ({computation.input.toFixed(4)}) = 1 / (1 + e^(-{computation.input.toFixed(4)}))
              </p>
              <div className="border-t border-slate-600 mt-3 pt-3">
                <p className="font-mono text-xl text-purple-400">
                  <strong>a = {computation.output.toFixed(4)}</strong>
                </p>
              </div>
            </div>
          </div>
        )}
        
        {computation?.type === 'complete' && (
          <div>
            <h3 className="text-2xl font-bold text-green-400 mb-4">✓ Feed Forward Complete!</h3>
            <div className="bg-slate-800 rounded p-4">
              <p className="text-lg text-slate-300 mb-3">Final Predictions:</p>
              <div className="flex gap-6">
                {computation.outputs.map((val, i) => (
                  <div key={i} className="bg-slate-600 rounded px-6 py-4">
                    <span className="text-slate-300 text-lg">y{i + 1} = </span>
                    <span className="font-bold text-green-400 text-2xl">{val.toFixed(4)}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
        
        {!computation && (
          <div className="text-slate-300 text-center py-8">
            <p className="text-xl mb-2">Ready to visualize the feed forward algorithm!</p>
            <p className="text-slate-400">Press Play to see each computation step by step</p>
          </div>
        )}
      </div>
      
      {/* Legend */}
      <div className="bg-slate-700 rounded-lg p-4 w-full max-w-5xl mb-6">
        <h4 className="text-lg font-bold text-white mb-3">Legend:</h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div className="flex items-center gap-2">
            <div className="w-12 h-6 bg-slate-800 border border-blue-400 rounded flex items-center justify-center text-blue-400 text-xs">0.25</div>
            <span className="text-slate-300">Weight (w)</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-purple-400 font-bold">b=0.1</span>
            <span className="text-slate-300">Bias</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-full border-2 border-blue-500 bg-slate-800"></div>
            <span className="text-slate-300">Neuron</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-12 h-0.5 bg-blue-500"></div>
            <span className="text-slate-300">Connection</span>
          </div>
        </div>
      </div>
      
      {/* Controls */}
      <div className="flex gap-4 mb-4">
        <button
          onClick={handlePrevStep}
          disabled={animationStep === 0}
          className="flex items-center gap-2 bg-slate-600 hover:bg-slate-700 text-white px-6 py-4 rounded-lg font-semibold transition-colors text-lg disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <ChevronLeft size={24} />
          Previous
        </button>
        <button
          onClick={handlePlayPause}
          className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white px-8 py-4 rounded-lg font-semibold transition-colors text-lg"
        >
          {isPlaying ? <Pause size={24} /> : <Play size={24} />}
          {isPlaying ? 'Pause' : animationStep >= MAX_STEPS ? 'Replay' : 'Play'}
        </button>
        <button
          onClick={handleNextStep}
          disabled={animationStep >= MAX_STEPS}
          className="flex items-center gap-2 bg-slate-600 hover:bg-slate-700 text-white px-6 py-4 rounded-lg font-semibold transition-colors text-lg disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Next
          <ChevronRight size={24} />
        </button>
        <button
          onClick={handleReset}
          className="flex items-center gap-2 bg-slate-600 hover:bg-slate-700 text-white px-6 py-4 rounded-lg font-semibold transition-colors text-lg"
        >
          <RotateCcw size={24} />
          Reset
        </button>
      </div>
      
      <div className="text-slate-400 text-sm mb-4">
        Step {animationStep} of {MAX_STEPS}
      </div>
      
      <div className="text-slate-400 text-sm text-center max-w-3xl">
        <p className="font-semibold mb-1">Each line shows the weight (w) that multiplies the input.</p>
        <p>Each neuron has a bias (b) shown to its left. The computation is: z = b + Σ(input × weight), then a = σ(z)</p>
      </div>
    </div>
  );
}