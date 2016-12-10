module Lib where

import Data.List(foldl', genericLength, genericTake, genericReplicate)

-- Length functions but with Num return
len :: Num a => [b] -> a
len = genericLength

-- Integral take
tak :: Integral a => a -> [b] -> [b]
tak = genericTake

-- Integral replicate
rep :: Integral a => a -> b -> [b]
rep = genericReplicate

-- Arithmetic Mean
mean :: Fractional a => [a] -> a
mean xs = sum xs / len xs

-- Apply two arguments, useful for zipWith3
apply2args :: (a -> b -> c) -> a -> b -> c
apply2args = (\x y z -> x y z)

-- Integral division
(//) :: (Integral a, Fractional b) => a -> a -> b
x // y = fromIntegral x / fromIntegral y

-- Euler's number
e :: Floating a => a
e = exp 1

