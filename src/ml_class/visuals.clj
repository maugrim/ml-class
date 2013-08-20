(ns ml-class.visuals
  (:require [ml-class.linear-regression :as lnr]
            [ml-class.logistic-regression :as lgr]
            [ml-class.gradient-descent :as gd]
            [ml-class.vector :as v]
            [clojure.data.csv :as csv]
            [clojure.java.io :as io])
  (:use [incanter core stats charts]))

(defn cost-plot [cost-fn vectors]
  (xy-plot (range 0 (count vectors)) (map cost-fn vectors)
           :title "Cost over time"
           :x-label "Iterations"
           :y-label "Cost"))

(defn pad [seq value]
  "Extends a sequence with an infinite stream of padding values."
  (concat seq (repeat value)))

(defn read-csv [path & {:keys [types] :or [[]]}]
  "Reads a CSV file at the given path. If a vector of :types converters is provided,
   having the same length as each row in the CSV file, then the converters are
   applied to each row by calling each converter with a value as an argument."
  (with-open [input (io/reader path)]
    (map (fn [values]
           (map (fn [conv val] (conv val)) (pad types identity) values))
         (doall (csv/read-csv input)))))

(defn read-dataset [path]
  "Reads a dataset at the given path, where each row is composed of N numbers;
   columns 0 through N-2 are interpreted as training feature vectors, and column
   N-1 is treated as a target value."
  (let [values (read-csv path :types (repeat read-string))]
    {:targets (v/vector (map last values))
     :features (map (comp v/vector butlast) values)}))
