(ns ml-class.gradient-descent
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
       (fn [& vals]
         (let [curr (nth (vec (trace vals)) i)
               next (assoc (vec vals) i (+ curr dx))]
           (/ (- (apply f next)
                 (apply f vals))
              dx))))))

(defn partial-derivatives
  "Return a seq of partial derivatives of a function of N variables
  n0, n1, n2... where the kth element in the list is the partial
  derivative of f with respect to nk."
  [f]
  (map (partial derivative f) (range (arg-count f))))

(defn step
  "Given a cost function J of one variable, a learning rate alpha,
  and a vector of parameter values, take a single gradient descent step and
  return a new vector of parameter values."
  [j alpha & vals]
  (let [derivatives (partial-derivatives j)
        step-value (fn [djdx x]
                     (- x (* alpha (apply djdx vals))))]
    (map step-value derivatives vals)))

(defn descend
  "Perform iterated gradient descent, starting with a vector of
  initial values and stopping when successive steps result in a
  difference of no more than epsilon."
  [j alpha epsilon & initial-vals]
  (let [descend-step (partial step j alpha)]
    (loop [curr-vals initial-vals]
      (let [curr-cost (trace "curr-cost" (apply j curr-vals))
            new-vals (trace "new-vals" (apply descend-step curr-vals))
            new-cost (trace "new-cost" (apply j new-vals))]
        (if (< (Math/abs (- curr-cost new-cost)) epsilon)
        new-vals
        (recur new-vals))))))
