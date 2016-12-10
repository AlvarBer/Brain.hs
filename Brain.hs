module Brain where

import Lib


type ActivFunc a = (a -> a)
type Neuron a = a -> [a] -> [a] -> a

--data Neuron' a = Neuron' {weights::[a], bias::a, act::a -> a}

{- This is the base neuron that given an activation function returns a neural
   network (Maybe this will be changed later) -}
neuron :: Num a => ActivFunc a -> Neuron a
neuron act bias weights inputs = act $ sum (zipWith (*) inputs weights) - bias


------------------ Activation functions -------------------
-- Step function
step :: (Num a, Ord a) => a -> a 
step signal = if signal > 0 then 1 else 0

-- Sigmoid function
sigmoid :: Floating a => a -> a 
sigmoid signal = 1 / (1 + e ** (-signal))


------------------- Articial Neurons ----------------------
{- Good ol' perceptron -}
perceptron :: (Num a, Ord a) => a -> [a] -> [a] -> a
perceptron = neuron step

{- sigmoidal neuron (with softer output, more suitable for learning) -}
sigmoidNeuron :: Floating a => a -> [a] -> [a] -> a
sigmoidNeuron = neuron sigmoid

{- simple NAND perceptron -}
nandPerceptron = perceptron 3 [-2, -2]


------------------ Layers and networks --------------------
{- generic neuron layer -}
layer :: (Num a, Integral b) => Neuron a -> b -> [a] -> [[a]] -> [a] -> [a]
layer neuron n biases weights ins = map ($ ins) $ zipWith3 (\x y z -> x y z) 
                                       (rep n neuron) biases weights

-- black means input “nodes”, white artificial neurons.
-- simpleNetwork layout (Not all connections shown):
-- ○-●
--  ✕ \
-- ○-●-●->
--  ✕ / 
-- ○-● 
simpleNetwork = sigmoidNeuron 0 [0, 0.5, 1] . hiddenLayer
        where hiddenLayer = layer sigmoidNeuron 3 biases weights
              biases = [0, 0, 0] 
              weights = [[0.3, 1, 0], [3, 1, 0], [0, 2, 0]]

--network :: [Layer] -> [a] -> [b]

-- Alt types
data Layer' a = Layer' [Neuron' a]
data NNetwork' a = NNetwork' [Layer' a]
data Neuron' a = Neuron' {act::(a -> a), bias::a, weights::[a]} 
