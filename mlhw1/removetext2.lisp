(defun removetext (word symbol)
       (mapcar #'(lambda (x) (if (equal x word) 
                                 'XXXXX 
                                 x
                              )
                  ) symbol)
 )