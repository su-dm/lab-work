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
  const layerX = { input: 150, hidden: 475, output: 800 };
  const getNodeY = (idx, total) => 100 + ((idx + 1) * 400 / (total + 1));
  
  const isWeightHighlighted = (from, fromIdx, to, toIdx) => {
    return highlightedWeights.some(w => 
      w.from === from && w.fromIdx === fromIdx && w.to === to && w.toIdx === toIdx
    );
  };
  
  return (
    <div className="w-full min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-800 flex flex-col items-center py-8 px-4">
      <div className="w-full max-w-7xl space-y-6">
        {/* Header */}
        <div className="text-center space-y-2 mb-8">
          <h1 className="text-4xl md:text-5xl font-bold text-white tracking-tight">
            Multi-Layer Perceptron
          </h1>
          <p className="text-lg text-slate-400">Feed Forward Algorithm Visualization</p>
        </div>

        {/* Network Visualization */}
        <div className="bg-slate-800/50 backdrop-blur-sm rounded-2xl shadow-2xl border border-slate-700/50 p-8 overflow-auto">
          <svg width="950" height="600" className="mx-auto">
            {/* Gradients and Filters */}
            <defs>
              {/* Input neurons gradient */}
              <linearGradient id="inputGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stopColor="#1e3a8a" stopOpacity="0.9" />
                <stop offset="100%" stopColor="#1e293b" stopOpacity="0.95" />
              </linearGradient>

              {/* Hidden neurons gradient */}
              <linearGradient id="hiddenGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stopColor="#581c87" stopOpacity="0.9" />
                <stop offset="100%" stopColor="#1e293b" stopOpacity="0.95" />
              </linearGradient>

              {/* Output neurons gradient */}
              <linearGradient id="outputGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stopColor="#065f46" stopOpacity="0.9" />
                <stop offset="100%" stopColor="#1e293b" stopOpacity="0.95" />
              </linearGradient>

              {/* Active neuron gradient */}
              <linearGradient id="activeGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stopColor="#1e40af" stopOpacity="1" />
                <stop offset="100%" stopColor="#1e3a8a" stopOpacity="1" />
              </linearGradient>

              {/* Glow filter */}
              <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
                <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
                <feMerge>
                  <feMergeNode in="coloredBlur"/>
                  <feMergeNode in="SourceGraphic"/>
                </feMerge>
              </filter>

              {/* Active glow filter */}
              <filter id="activeGlow" x="-50%" y="-50%" width="200%" height="200%">
                <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
                <feMerge>
                  <feMergeNode in="coloredBlur"/>
                  <feMergeNode in="SourceGraphic"/>
                </feMerge>
              </filter>
            </defs>

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
                        x={midX - 20}
                        y={midY - 9}
                        width="40"
                        height="18"
                        fill="#0f172a"
                        stroke="#334155"
                        strokeWidth="1"
                        rx="4"
                        opacity={0.85}
                      />
                      <text
                        x={midX}
                        y={midY + 4}
                        fill="#cbd5e1"
                        fontSize="10"
                        textAnchor="middle"
                        fontWeight="500"
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
                        x={midX - 20}
                        y={midY - 9}
                        width="40"
                        height="18"
                        fill="#0f172a"
                        stroke="#334155"
                        strokeWidth="1"
                        rx="4"
                        opacity={0.85}
                      />
                      <text
                        x={midX}
                        y={midY + 4}
                        fill="#cbd5e1"
                        fontSize="10"
                        textAnchor="middle"
                        fontWeight="500"
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
          <text x={layerX.input} y="50" fill="#cbd5e1" fontSize="15" fontWeight="600" textAnchor="middle">Input Layer</text>
          {inputs.map((val, i) => (
            <g key={`input-${i}`}>
              <circle
                cx={layerX.input}
                cy={getNodeY(i, 3)}
                r="32"
                fill="url(#inputGradient)"
                stroke="#3b82f6"
                strokeWidth="2.5"
                filter="url(#glow)"
              />
              <text x={layerX.input} y={getNodeY(i, 3) - 6} fill="#93c5fd" fontSize="11" textAnchor="middle" fontWeight="600">
                {inputLabels[i]}
              </text>
              <text x={layerX.input} y={getNodeY(i, 3) + 10} fill="#dbeafe" fontSize="17" fontWeight="bold" textAnchor="middle">
                {val.toFixed(1)}
              </text>
            </g>
          ))}
          
          {/* Hidden Layer */}
          <text x={layerX.hidden} y="50" fill="#cbd5e1" fontSize="15" fontWeight="600" textAnchor="middle">Hidden Layer (Sigmoid)</text>
          {Array(4).fill(0).map((_, i) => {
            const isActive = activeNeuron?.layer === 'hidden' && activeNeuron?.index === i;
            const showValue = animationStep > i * 2 + 1;
            return (
              <g key={`hidden-${i}`}>
                <circle
                  cx={layerX.hidden}
                  cy={getNodeY(i, 4)}
                  r="32"
                  fill={isActive ? "url(#activeGradient)" : "url(#hiddenGradient)"}
                  stroke={isActive ? "#60a5fa" : "#a78bfa"}
                  strokeWidth={isActive ? "3" : "2.5"}
                  filter={isActive ? "url(#activeGlow)" : "url(#glow)"}
                  className="transition-all duration-300"
                />
                {/* Bias label */}
                <text
                  x={layerX.hidden - 55}
                  y={getNodeY(i, 4) + 4}
                  fill="#c4b5fd"
                  fontSize="11"
                  fontWeight="600"
                >
                  b={biasHidden[i].toFixed(2)}
                </text>
                <text x={layerX.hidden} y={getNodeY(i, 4) - 6} fill="#c4b5fd" fontSize="11" textAnchor="middle" fontWeight="600">
                  h{i+1}
                </text>
                {showValue && (
                  <text x={layerX.hidden} y={getNodeY(i, 4) + 10} fill="#e9d5ff" fontSize="16" fontWeight="bold" textAnchor="middle">
                    {hidden[i].toFixed(2)}
                  </text>
                )}
              </g>
            );
          })}
          
          {/* Output Layer */}
          <text x={layerX.output} y="50" fill="#cbd5e1" fontSize="15" fontWeight="600" textAnchor="middle">Output Layer</text>
          {Array(2).fill(0).map((_, i) => {
            const isActive = activeNeuron?.layer === 'output' && activeNeuron?.index === i;
            const showValue = animationStep > 8 + i * 2 + 1;
            return (
              <g key={`output-${i}`}>
                <circle
                  cx={layerX.output}
                  cy={getNodeY(i, 2)}
                  r="32"
                  fill={isActive ? "url(#activeGradient)" : "url(#outputGradient)"}
                  stroke={isActive ? "#60a5fa" : "#34d399"}
                  strokeWidth={isActive ? "3" : "2.5"}
                  filter={isActive ? "url(#activeGlow)" : "url(#glow)"}
                  className="transition-all duration-300"
                />
                {/* Bias label */}
                <text
                  x={layerX.output - 55}
                  y={getNodeY(i, 2) + 4}
                  fill="#6ee7b7"
                  fontSize="11"
                  fontWeight="600"
                >
                  b={biasOutput[i].toFixed(2)}
                </text>
                <text x={layerX.output} y={getNodeY(i, 2) - 6} fill="#6ee7b7" fontSize="11" textAnchor="middle" fontWeight="600">
                  y{i+1}
                </text>
                {showValue && (
                  <text x={layerX.output} y={getNodeY(i, 2) + 10} fill="#d1fae5" fontSize="17" fontWeight="bold" textAnchor="middle">
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
                    strokeWidth="1.5"
                    rx="5"
                  />
                  <text
                    x={midX}
                    y={midY + 4}
                    fill="#dbeafe"
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
                    strokeWidth="1.5"
                    rx="5"
                  />
                  <text
                    x={midX}
                    y={midY + 4}
                    fill="#dbeafe"
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
        <div className="bg-slate-800/50 backdrop-blur-sm rounded-2xl shadow-xl border border-slate-700/50 overflow-hidden">
          {computation?.type === 'weighted_sum' && (
            <div className="p-8">
              {/* Step Header */}
              <div className="mb-6">
                <div className="inline-block px-4 py-1.5 bg-blue-500/20 border border-blue-500/30 rounded-full mb-3">
                  <span className="text-blue-300 text-sm font-semibold">Step {animationStep + 1} of {MAX_STEPS}</span>
                </div>
                <h3 className="text-2xl font-bold text-white">
                  Computing {computation.layer === 'hidden' ? 'Hidden' : 'Output'} Neuron {computation.neuron + 1}
                </h3>
              </div>

              {/* Formula Section */}
              <div className="bg-gradient-to-br from-slate-900/80 to-slate-800/80 rounded-xl p-6 mb-4 border border-slate-700/30">
                <p className="text-sm font-semibold text-emerald-400 mb-3 uppercase tracking-wide">Weighted Sum Formula</p>
                <p className="text-2xl font-mono text-white">z = b + Σ(xᵢ × wᵢ)</p>
              </div>

              {/* Calculation Section */}
              <div className="bg-gradient-to-br from-slate-900/80 to-slate-800/80 rounded-xl p-6 border border-slate-700/30">
                <p className="text-sm font-semibold text-amber-400 mb-4 uppercase tracking-wide">Step-by-Step Calculation</p>
                <div className="space-y-2 mb-4">
                  {computation.terms.map((term, idx) => (
                    <div key={idx} className="flex items-center font-mono text-sm text-slate-300">
                      <span className="text-slate-500 mr-3">{idx === 0 ? '=' : '+'}</span>
                      <span>{term}</span>
                    </div>
                  ))}
                </div>
                <div className="border-t border-slate-600/50 pt-4 mt-4">
                  <div className="inline-flex items-center gap-3 bg-amber-500/10 px-4 py-2.5 rounded-lg border border-amber-500/20">
                    <span className="text-slate-400 text-sm">Result:</span>
                    <span className="font-mono text-xl font-bold text-amber-400">z = {computation.total.toFixed(4)}</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {computation?.type === 'activation' && (
            <div className="p-8">
              {/* Step Header */}
              <div className="mb-6">
                <div className="inline-block px-4 py-1.5 bg-purple-500/20 border border-purple-500/30 rounded-full mb-3">
                  <span className="text-purple-300 text-sm font-semibold">Step {animationStep + 1} of {MAX_STEPS}</span>
                </div>
                <h3 className="text-2xl font-bold text-white">
                  Applying Activation Function
                </h3>
              </div>

              {/* Formula Section */}
              <div className="bg-gradient-to-br from-slate-900/80 to-slate-800/80 rounded-xl p-6 mb-4 border border-slate-700/30">
                <p className="text-sm font-semibold text-emerald-400 mb-3 uppercase tracking-wide">Sigmoid Function</p>
                <p className="text-2xl font-mono text-white">σ(z) = 1 / (1 + e^(-z))</p>
              </div>

              {/* Calculation Section */}
              <div className="bg-gradient-to-br from-slate-900/80 to-slate-800/80 rounded-xl p-6 border border-slate-700/30">
                <p className="text-sm font-semibold text-amber-400 mb-4 uppercase tracking-wide">Calculation</p>
                <div className="font-mono text-sm text-slate-300 mb-4">
                  σ({computation.input.toFixed(4)}) = 1 / (1 + e^(-{computation.input.toFixed(4)}))
                </div>
                <div className="border-t border-slate-600/50 pt-4 mt-4">
                  <div className="inline-flex items-center gap-3 bg-purple-500/10 px-4 py-2.5 rounded-lg border border-purple-500/20">
                    <span className="text-slate-400 text-sm">Activation:</span>
                    <span className="font-mono text-xl font-bold text-purple-400">a = {computation.output.toFixed(4)}</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {computation?.type === 'complete' && (
            <div className="p-8">
              {/* Completion Header */}
              <div className="mb-6 text-center">
                <div className="inline-flex items-center gap-3 px-5 py-2 bg-emerald-500/20 border border-emerald-500/30 rounded-full mb-4">
                  <svg className="w-5 h-5 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  <span className="text-emerald-300 font-semibold">Complete</span>
                </div>
                <h3 className="text-2xl font-bold text-white">Feed Forward Complete</h3>
              </div>

              {/* Results */}
              <div className="bg-gradient-to-br from-slate-900/80 to-slate-800/80 rounded-xl p-6 border border-slate-700/30">
                <p className="text-sm font-semibold text-emerald-400 mb-4 uppercase tracking-wide">Final Output Predictions</p>
                <div className="flex flex-wrap justify-center gap-4">
                  {computation.outputs.map((val, i) => (
                    <div key={i} className="bg-gradient-to-br from-emerald-500/10 to-emerald-600/10 border border-emerald-500/20 rounded-xl px-6 py-4 min-w-[180px]">
                      <div className="text-center">
                        <div className="text-slate-400 text-sm mb-1">Output {i + 1}</div>
                        <div className="font-mono text-2xl font-bold text-emerald-400">y{i + 1} = {val.toFixed(4)}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {!computation && (
            <div className="p-12 text-center">
              <div className="max-w-md mx-auto space-y-3">
                <div className="w-16 h-16 mx-auto bg-blue-500/20 rounded-full flex items-center justify-center mb-4">
                  <Play size={32} className="text-blue-400" />
                </div>
                <h3 className="text-xl font-semibold text-white">Ready to Begin</h3>
                <p className="text-slate-400">Press <span className="text-blue-400 font-semibold">Play</span> to visualize each step of the feed forward algorithm</p>
              </div>
            </div>
          )}
        </div>

        {/* Legend */}
        <div className="bg-slate-800/50 backdrop-blur-sm rounded-2xl shadow-xl border border-slate-700/50 p-6">
          <h4 className="text-sm font-semibold text-slate-400 uppercase tracking-wide mb-5">Legend</h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            <div className="flex items-center gap-3">
              <div className="w-14 h-7 bg-slate-900/50 border border-blue-400/40 rounded-lg flex items-center justify-center text-blue-400 text-xs font-semibold shadow-sm">0.25</div>
              <span className="text-slate-300 text-sm">Weight (w)</span>
            </div>
            <div className="flex items-center gap-3">
              <span className="text-purple-400 font-bold text-sm px-2">b=0.1</span>
              <span className="text-slate-300 text-sm">Bias</span>
            </div>
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-full border-2 border-blue-500 bg-slate-900/50 shadow-sm"></div>
              <span className="text-slate-300 text-sm">Neuron</span>
            </div>
            <div className="flex items-center gap-3">
              <div className="w-14 h-1 bg-blue-500 rounded-full shadow-sm"></div>
              <span className="text-slate-300 text-sm">Connection</span>
            </div>
          </div>
        </div>

        {/* Controls */}
        <div className="bg-slate-800/50 backdrop-blur-sm rounded-2xl shadow-xl border border-slate-700/50 p-6">
          <div className="flex flex-col items-center gap-4">
            {/* Button Group */}
            <div className="flex flex-wrap justify-center gap-3">
              <button
                onClick={handlePrevStep}
                disabled={animationStep === 0}
                className="group flex items-center gap-2 bg-slate-700 hover:bg-slate-600 text-white px-6 py-3 rounded-xl font-medium transition-all shadow-lg disabled:opacity-30 disabled:cursor-not-allowed disabled:hover:bg-slate-700 border border-slate-600 hover:border-slate-500"
              >
                <ChevronLeft size={20} className="group-hover:-translate-x-0.5 transition-transform" />
                <span>Previous</span>
              </button>

              <button
                onClick={handlePlayPause}
                className="group flex items-center gap-2 bg-gradient-to-br from-blue-600 to-blue-700 hover:from-blue-500 hover:to-blue-600 text-white px-8 py-3 rounded-xl font-semibold transition-all shadow-lg hover:shadow-blue-500/25 border border-blue-500/50"
              >
                {isPlaying ? <Pause size={20} className="group-hover:scale-110 transition-transform" /> : <Play size={20} className="group-hover:scale-110 transition-transform" />}
                <span>{isPlaying ? 'Pause' : animationStep >= MAX_STEPS ? 'Replay' : 'Play'}</span>
              </button>

              <button
                onClick={handleNextStep}
                disabled={animationStep >= MAX_STEPS}
                className="group flex items-center gap-2 bg-slate-700 hover:bg-slate-600 text-white px-6 py-3 rounded-xl font-medium transition-all shadow-lg disabled:opacity-30 disabled:cursor-not-allowed disabled:hover:bg-slate-700 border border-slate-600 hover:border-slate-500"
              >
                <span>Next</span>
                <ChevronRight size={20} className="group-hover:translate-x-0.5 transition-transform" />
              </button>

              <button
                onClick={handleReset}
                className="group flex items-center gap-2 bg-slate-700 hover:bg-slate-600 text-white px-6 py-3 rounded-xl font-medium transition-all shadow-lg border border-slate-600 hover:border-slate-500"
              >
                <RotateCcw size={20} className="group-hover:rotate-180 transition-transform duration-500" />
                <span>Reset</span>
              </button>
            </div>

            {/* Progress Indicator */}
            <div className="flex flex-col items-center gap-3 w-full max-w-md">
              <div className="w-full bg-slate-700/50 rounded-full h-2 overflow-hidden">
                <div
                  className="bg-gradient-to-r from-blue-500 to-purple-500 h-full transition-all duration-300 rounded-full"
                  style={{ width: `${(animationStep / MAX_STEPS) * 100}%` }}
                />
              </div>
              <div className="text-slate-400 text-sm font-medium">
                Step {animationStep} of {MAX_STEPS}
              </div>
            </div>

            {/* Info Text */}
            <div className="text-center text-slate-400 text-sm max-w-2xl pt-2 border-t border-slate-700/50">
              <p className="leading-relaxed">
                Each connection displays its weight (w). Biases (b) appear next to neurons.
                Computation: <span className="text-slate-300 font-mono">z = b + Σ(input × weight)</span>, then <span className="text-slate-300 font-mono">a = σ(z)</span>
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}