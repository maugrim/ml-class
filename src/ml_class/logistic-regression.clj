(ns ml-class.logistic-regression
  (:use [ml-class.math])
  (:require [ml-class.gradient-descent :as gd]
            [ml-class.matrix :as m]
            [ml-class.vector :as v]))

(defn distance
  "Cost(x, y) = -y log x - (1-y) log x"
  [value target]
  (- (- (* target (Math/log value))) (* (- 1 target) (Math/log value))))

(defn distances [values targets]
  (reduce + (map distance values targets)))

(defn logistic [z]
  (/ 1 (+ 1 (Math/exp (- z)))))

(def hypothesis (comp logistic v/dot-product))

(defn cost-fn
  "Given a set of training vectors and their targets, generates a cost
  function of a parameter vector theta which quantifies the error against
  the targets when using that parameter vector."
  [features targets]
  (let [vectors (map #(cons 1 %) features)]
    (fn [& theta]
      (average (distances (map #(hypothesis theta %) vectors) targets)))))

(defn logistic-regression
  "Performs logistic regression on the training set with the specified parameters."
  [alpha initial training-vectors targets]
  (gd/run (cost-fn training-vectors targets) alpha initial))
