(ns ml-class.vector)

(def vector clojure.core/vec)
(def vector? clojure.core/vector?)

(defn plus [& vs]
  (apply mapv + vs))

(defn minus [& vs]
  (apply mapv - vs))

(defn scalar-op [v op & args]
  (mapv #(reduce op % args) v))

(defn mult [scalar v]
  (scalar-op v * scalar))

(defn div [scalar v]
  (scalar-op v / scalar))

(defn dot-product [a b]
  (reduce + (map * a b)))

(defn magnitude [v]
  (Math/sqrt (reduce + (map #(* % %) v))))
