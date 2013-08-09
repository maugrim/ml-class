(ns ml-class.math)

(defn square [x] (* x x))

(defn average [& values]
  (/ (apply + values) (count values)))
