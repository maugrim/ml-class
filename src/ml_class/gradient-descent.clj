(ns ml-class.gradient-descent)

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
  "Given a cost function J taking some parameters, a learning rate alpha,
  and a vector of parameter values, take a single gradient descent step and
  return a new vector of parameter values."
  [j alpha theta]
  (let [derivatives (partial-derivatives j)
        step-value (fn [djdx x]
                     (- x (* alpha (apply djdx theta))))]
    (map step-value derivatives theta)))

(defn descend
  "Perform iterated gradient descent, starting with a vector of
  initial values and stopping when successive steps result in a
  difference of no more than epsilon."
  [j alpha epsilon initial]
  (let [descend-step (partial step j alpha)]
    (loop [curr-theta initial]
      (let [curr-cost (trace "curr-cost" (apply j curr-theta))
            new-theta (trace "new-vals" (descend-step curr-theta))
            new-cost (trace "new-cost" (apply j new-theta))]
        (if (< (Math/abs (- curr-cost new-cost)) epsilon)
        new-theta
        (recur new-theta))))))
