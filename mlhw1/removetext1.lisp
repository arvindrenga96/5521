(defun removetext (word symbol)
	(cond
		((null symbol) nil)
		((eql word (car symbol))
			(cons 'ZZZZ (removetext word (cdr symbol)))
		)
		
		(t (cons
		     	(car symbol)
		     	(removetext word (cdr symbol) ) )
		)
	)
)