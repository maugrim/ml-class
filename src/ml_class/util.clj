(ns ml-class.util
  (:require [clojure.data.csv :as csv]
            [clojure.java.io :as io]))

(defn square [x] (* x x))

(defn average [values]
  (float (/ (reduce + values) (count values))))

(defn distance [value target]
  (square (- value target)))

(defn sd [values]
  (let [mean (average values)
	distances (map (partial distance mean) values)]
    (Math/sqrt (average distances))))

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
    {:targets (mapv last values)
     :features (mapv (comp vec butlast) values)}))
