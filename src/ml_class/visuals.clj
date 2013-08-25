(ns ml-class.visuals
  (:require [ml-class.linear-regression :as lnr]
            [ml-class.logistic-regression :as lgr]
            [ml-class.gradient-descent :as gd]
            [ml-class.vector :as v])
  (:use [incanter core stats charts]
        [ml-class.util]))

(defn cost-plot [cost-fn vectors]
  (xy-plot (range 0 (count vectors)) (map cost-fn vectors)
           :title "Cost over time"
           :x-label "Iterations"
           :y-label "Cost"))
