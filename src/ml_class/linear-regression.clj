(ns ml-class.linear-regression
  (:use [clojure.tools.trace])
  (:require [ml-class.gradient-descent :as gd])
  (:require [ml-class.matrix :as m])
  (:require [ml-class.vector :as v]))

(defn square [x] (* x x))

(defn distance [values targets]
  (apply + (map #(square (- %1 %2)) values targets)))

(defn average [& values]
  (/ (apply + values) (count values)))

(defn cost-fn
  "Given a set of training vectors and their targets, generates a cost
  function of a parameter vector theta which computes the average
  distance to the targets when using that parameter vector."
  [features targets]
  (let [vectors (map #(cons 1 %) features)]
    (fn [& theta]
      (/ 2 (average (map (partial v/dot-product theta) vectors))))))

(defn linear-regression
  "Performs linear regression on the training set with the specified parameters."
  [alpha epsilon initial training-vectors targets]
  (gd/descend (cost-fn training-vectors targets) alpha epsilon initial))
