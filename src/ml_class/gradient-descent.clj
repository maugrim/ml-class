(ns ml-class.gradient-descent
  (:use [clojure.math.numeric-tower])
  (:use [clojure.tools.trace]))

(defn derivative
  "Return the derivative of a function of one variable."
  [f]
  (let [dx 0.00000001]
    (fn [x]
      (/ (- (f (+ x dx))
            (f x))
         dx))))

(defn step
  "Given a cost function J of one variable, a current value theta, and
  a learning rate alpha, take a single gradient descent step and
  return a new theta."
  [j alpha theta]
  (let [dj (derivative j)]
    (- theta (* alpha (dj theta)))))

(defn descend
  "Perform iterated gradient descent, starting with an initial value
  and stopping when successive steps result in a difference of no more
  than epsilon."
  [j alpha initial epsilon]
  (loop [theta initial]
    (let [theta-prime (step j alpha theta)]
      (if (< (abs (- theta theta-prime)) epsilon)
        theta-prime
        (recur theta-prime)))))
