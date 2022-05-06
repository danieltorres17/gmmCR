def bmatrix(a, transpose=False):
  """
  Returns a LaTeX bmatrix
  a: numpy array
  :returns: LaTeX bmatrix as a string

  From:
  https://stackoverflow.com/questions/17129290/numpy-2d-and-1d-array-to-latex-bmatrix/17131750
  """
  if len(a.shape) > 2:
      raise ValueError('bmatrix can at most display two dimensions')
  lines = str(a).replace('[', '').replace(']', '').splitlines()
  rv = [r'\begin{bmatrix}']
  if not transpose:
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
  else:
    rv += ['  ' + ' \\ '.join(l.split()) + r'\\' for l in lines]
  rv +=  [r'\end{bmatrix}']

  return '\n'.join(rv)