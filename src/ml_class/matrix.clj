(ns ml-class.matrix
  (:use clojure.math.combinatorics)
  (:require [ml-class.vector :as v]))

(def trans (partial apply mapv vector))

(defn row [matrix i]
  (nth matrix i))

(defn col [matrix i]
  (row (trans matrix) i))

(defn plus [& matrices]
  (apply mapv v/plus matrices))

(defn minus [& matrices]
  (apply mapv v/minus matrices))

(defn scalar-op [m op & args]
  (mapv #(apply v/scalar-op % op args) m))

(defn mult [scalar m]
  (scalar-op m * scalar))

(defn div [scalar m]
  (scalar-op m / scalar))
