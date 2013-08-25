(ns ml-class.linear-regression
  (:use [ml-class.util])
  (:require [ml-class.gradient-descent :as gd]
            [ml-class.matrix :as m]
            [ml-class.vector :as v]))

(defn distance [value target]
  (square (- value target)))

(defn hypothesis-fn
  "Returns a hypothesis function for a vector x, given a parameter vector theta."
  [theta]
  (fn [x]
    (v/dot-product (vec (cons 1 x)) theta)))

(defn cost-fn
  "Given a set of training vectors and their targets, generates a cost
  function of a parameter vector theta which quantifies the error against
  the targets when using that parameter vector."
  [features targets]
  (fn [theta]
    (let [hypothesis (hypothesis-fn theta)]
      (/ (average (map distance (map hypothesis features) targets)) 2))))

(defn run
  "Performs linear regression on the training set with the specified parameters."
  [training-vectors targets & {:keys [alpha initial n]
                               :or {alpha 0.01 initial [1 1] n 500}}]
  (let [j (cost-fn training-vectors targets)
        results (take n (gd/run j alpha initial))]
    {:iterations (vec results)
     :hypotheses (map hypothesis-fn results)
     :costs (map j results)
     :theta (last results)
     :j j}))
