(ns ml-class.gradient-descent
  (:use [ml-class.math]))

(defn arg-count [f]
  (let [m (first (.getDeclaredMethods (class f)))
        p (.getParameterTypes m)]
    (alength p)))

(defn scale-fn
  "Given some feature values, return an appropriate scaling function
  to scale and mean-normalize the feature in question."
  [& vals]
  (let [mean (apply average vals)
        range (- (apply max vals) (apply min vals))]
    (fn [x] (float (/ (- x mean) range)))))

(defn derivative
  "Returns the partial derivative of a function f with respect to its
  ith parameter."
  ([f] (derivative f 0))
  ([f i]
     (let [dx 0.0000001]
       (fn [& vals]
         (let [curr (nth (vec vals) i)
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
  (let [step-value (fn [djdx x]
                     (- x (* alpha (apply djdx theta))))]
    (map step-value (partial-derivatives j) theta)))

(defn run
  "Perform iterated gradient descent, starting with a vector of
  initial values. Creates an infinite sequence of successive vectors."
  [j alpha initial]
  (iterate (partial step j alpha) initial))
