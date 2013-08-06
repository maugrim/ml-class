(ns ml-class.gradient-descent
  (:use [clojure.math.numeric-tower])
  (:use [clojure.tools.trace]))

(defn arg-count [f]
  (let [m (first (.getDeclaredMethods (class f)))
        p (.getParameterTypes m)]
    (alength p)))

(defn derivative
  "Returns the partial derivative of a function f with respect to its
  ith parameter."
  ([f] (derivative f 0))
  ([f i]
     (let [dx 0.0000001]
       (fn [& vars]
         (let [curr (nth vars i)
               next (assoc (vec vars) i (+ curr dx))]
           (/ (- (apply f next)
                 (apply f vars))
              dx))))))

(defn partial-derivatives
  "Return a seq of partial derivatives of a function of N variables
  n0, n1, n2... where the kth element in the list is the partial
  derivative of f with respect to nk."
  [f]
  (map (partial derivative f) (range (arg-count f))))

(defn step
  "Given a cost function J of one variable, a current value theta, and
  a learning rate alpha, take a single gradient descent step and
  return a new theta."
  [j alpha theta]
  (let [dj (derivative j)]
    (- theta (* alpha (dj theta)))))

(defn descend
  "Perform iterated gradient descent, starting with an initial value
  and stopping when successive steps result in a difference of no more
  than epsilon."
  [j alpha initial epsilon]
  (loop [theta initial]
    (let [theta-prime (step j alpha theta)]
      (if (< (abs (- theta theta-prime)) epsilon)
        theta-prime
        (recur theta-prime)))))
