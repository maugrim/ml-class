(ns ml-class.matrix
  (:use clojure.math.combinatorics)
  (:require [ml-class.vector :as v]))

(def trans (partial apply mapv vector))
(def rows seq)
(def cols trans)

(defn row [matrix i]
  (nth (rows matrix) i))

(defn col [matrix i]
  (nth (cols matrix) i))

(def matrix vector) ; a matrix is a Clojure vector of vectors
(defn matrix? [x]
  (and (v/vector? x) (every? v/vector? (rows x))))

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

(defn identity-matrix
  "Generates an identity matrix of size n."
  [n]
  (letfn [(identity-row [i] (conj (repeat (dec i) 0) 1))
          (row-generator [n] (partition n (dec n) (cycle (identity-row n))))]
    (apply matrix (take n (map (partial apply v/vector) (row-generator n))))))

(defn mmult
  "Multiplies the matrices a and b."
  [a b]
  (mapv (fn [row]
          (mapv (fn [col]
                  (v/dot-product row col))
                (cols b)))
        (rows a)))
