(ns ml-class.gradient-descent
  (:use [ml-class.util]))

(defn scale-fn
  "Given some feature values, return an appropriate scaling function
  to scale and mean-normalize the feature in question."
  [& vals]
  (let [mean (apply average vals)
        range (- (apply max vals) (apply min vals))]
    (fn [x] (float (/ (- x mean) range)))))

(defn derivative
  "Returns the partial derivative of a function f which takes a
  vector [x0, x1...xn] with respect to its ith parameter."
  ([f] (derivative f 0))
  ([f i]
     (let [dx 0.0000001]
       (fn [vals]
         (let [curr (nth vals i)
               next (assoc vals i (+ curr dx))]
           (/ (- (f next) (f vals)) dx))))))

(defn partial-derivatives
  "Return a seq of partial derivatives of a function f which takes
  a vector [x0, x1...xn] where the ith element in the list is the
  partial derivative of f with respect to xi."
  [f n]
  (map (partial derivative f) (range n)))

(defn step
  "Given a cost function J taking a parameter vector, a learning rate alpha,
  and a vector of parameter values, take a single gradient descent step and
  return a new vector of parameter values."
  [j alpha theta]
  (let [step-value (fn [djdx x]
                     (- x (* alpha (djdx theta))))]
    (mapv step-value (partial-derivatives j (count theta)) theta)))

(defn run
  "Perform iterated gradient descent, starting with a vector of
  initial values. Creates an infinite sequence of successive vectors."
  [j alpha initial]
  (iterate (partial step j alpha) initial))
