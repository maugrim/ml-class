(ns ml-class.vector)

(def vector clojure.core/vector)
(def vector? clojure.core/vector?)

(defn plus [& vs]
  (apply mapv + vs))

(defn minus [& vs]
  (apply mapv - vs))

(defn scalar-op [v op & args]
  (mapv #(apply op % args) v))

(defn mult [scalar v]
  (scalar-op v * scalar))

(defn div [scalar v]
  (scalar-op v / scalar))

(defn dot-product [a b]
  (apply + (map * a b)))
