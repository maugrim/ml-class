(ns ml-class.linear-regression
  (:use [ml-class.math])
  (:require [ml-class.gradient-descent :as gd]
            [ml-class.matrix :as m]
            [ml-class.vector :as v]))

(defn distance [value target]
  (square (- value target)))

(defn distances [values targets]
  (apply + (map distance values targets)))

(def hypothesis v/dot-product)

(defn cost-fn
  "Given a set of training vectors and their targets, generates a cost
  function of a parameter vector theta which quantifies the error against
  the targets when using that parameter vector."
  [features targets]
  (let [vectors (map #(cons 1 %) features)]
    (fn [& theta]
      (/ 2 (average (distances (map #(hypothesis theta %) vectors)))))))

(defn linear-regression
  "Performs linear regression on the training set with the specified parameters."
  [alpha initial training-vectors targets]
  (gd/run (cost-fn training-vectors targets) alpha initial))
