# function_as_script.py:
# Execute a Python function as if it were part of a script.

# How to use:
#  * Keep variable names in script EXACTLY matching
#    variable names in function.
#  * Assign parameters in script
#  *   from function_as_script import scriptify
#  *   scriptified_function=scriptify(my_function)
#  * Call scriptified_function with parameters
#  * Can supply keyword arguments to cause assignments
#    from globals of different names. The assignments
#    occur with each script call in the __main__
#    namespace
#  * Default arguments work, but relying on them is
#    dangerous if you have calls with/without the
#    default arguments in the script, because
#    the values assigned in one call will be
#    reused in the next call (because they are
#    now in the __main__ namespace) 
#  * Note that local variables assigned in the
#    function WILL OVERRIDE VARIABLES IN THE
#    __main__ NAMESPACE! These changes will
#    persiste even after the function returns!!!
#  * After calling scriptify() all global variables
#    from the function's module will be mapped into
#    the __main__ namespace. scriptify() will
#    fail if there is a namespace conflict.
#
# WARNING: For the moment, if the function depends on
# __future__ statements, those may cause the
# scriptification to fail!

import sys
import inspect
import ast

def add_to_lineos(astobj,lineno_increment):
    # (NO_LONGER_NEEDED!)
    # Increment this object
    if hasattr(astobj,"lineno"):
        astobj.lineno += lineno_increment
        pass

    # recursively increment linenumbers for all fields
    for field in astobj._fields:
        if hasattr(astobj,field):
            field_data = getattr(astobj,field)

            # if the field is a list
            if isinstance(field_data,list):
                # Operate on each elemetn of list
                [ add_to_linenos(field_datum) for field_datum in field_data ]
                pass
            else:
                # otherwise the field should be an AST object
                add_to_linenos(field_data)
                pass
            
            pass
        pass
    pass

    

def scriptify(callable):

    codeobj = callable.__code__

    #(sourcelines,firstlineno) = inspect.getsourcelines(callable)
    #sourcecode = "".join(sourcelines)

    sourcefile=inspect.getsourcefile(callable)
    sourcecode=open(sourcefile,"r").read()
    

    syntree = ast.parse(sourcecode)
    assert(syntree.__class__ is ast.Module)

    # Find the correct function definition
    synsubtree=None
    for BodyEl in syntree.body:
        if isinstance(BodyEl,ast.FunctionDef):
            # Found a function definition
            if BodyEl.name==callable.__name__:
                # Found our function
                synsubtree = BodyEl
                break
            pass
        pass

    if synsubtree is None:
        raise ValueError("Definition of function %s not found in %s" % (callable.__name__,sourcefile))
    
    
    ## Iterate over syntree, correcting line numbers
    #add_to_linenos(syntree,firstlineno-1)

    #assert(syntree.body[0].__class__ is ast.FunctionDef)

    args = synsubtree.args.args
    argnames = [ arg.id for arg in args ]
    
    # Extract default arguments
    assert((len(synsubtree.args.defaults) == 0 and callable.__defaults__ is None) or len(synsubtree.args.defaults) == len(callable.__defaults__))

    
    mandatoryargnames = argnames[:(len(argnames)-len(synsubtree.args.defaults))]
    defaultargnames = argnames[(len(argnames)-len(synsubtree.args.defaults)):]

    context = sys.modules["__main__"].__dict__

    CodeModule=ast.Module(body=synsubtree.body,lineno=0)

    # if CodeModule ends with return statement, replace with
    # assignment to _fas_returnval
    has_return=False
    if isinstance(CodeModule.body[-1],ast.Return) and CodeModule.body[-1].value is not None:
        has_return=True
        new_assignment = ast.Assign(targets=[ast.Name(id="_fas_returnval",lineno=CodeModule.body[-1].lineno,col_offset=0,ctx=ast.Store())],value=CodeModule.body[-1].value,lineno=CodeModule.body[-1].lineno,col_offset=0)
        CodeModule.body.pop()
        CodeModule.body.append(new_assignment)
        pass
    
    

    codeobj = compile(CodeModule,inspect.getsourcefile(callable),"exec",dont_inherit=True)  # should be able to set flags based on __future__ statments in original source module, but we don't currently do this
    
    def scriptified(*args,**kwargs):
        # Pass arguments
        for argcnt in range(len(mandatoryargnames)):
            argname=mandatoryargnames[argcnt]
            if argcnt < len(args):
                argvalue = args[argcnt]
                pass
            elif argname in kwargs:
                argvalue=kwargs[argname]
                pass
            else:
                raise ValueError("Argument %s must be provided" % (argname))

            context[argname] = argvalue
            pass

        # Optional arguments
        for argcnt in range(len(defaultargnames)):
            argname=defaultargnames[argcnt]
            if argcnt+len(mandatoryargnames) < len(args):
                argvalue = args[argcnt+len(mandatoryargnames)]
                pass
            elif argname in kwargs:
                argvalue=kwargs[argname]
                pass
            else:
                # default value
                argvalue=callable.__defaults__[argcnt]
                pass
                
            context[argname] = argvalue
            pass
        
 
        #
        ## Assign default arguments where needed        
        #for defaultargnum in range(len(defaultargnames)):
        #    argname=defaultargnames[defaultargnum]
        #    if not argname in context and not argname in kwargs:
        #        context[argname] = callable.__defaults__[defaultargnum]
        #        pass
        #    pass
        
        # execute!
        exec(codeobj,context,context)
        if has_return:
            retval=context["_fas_returnval"]
            pass
        else:
            retval=None
        return retval
    

    # Add globals from function module into __main__ namespace
    if hasattr(callable,"__globals__"):
        globaldict=callable.__globals__
        pass
    else:
        globaldict=sys.modules[callable.__module__].__dict__
        pass

    for varname in globaldict:
        if varname.startswith("__"):
            # do not transfer any variables starting with "__"
            pass
        elif varname in context:
            if context[varname].__name__=="scriptified":
                # Do not object to (but do not overwrite)
                # a scriptified function
                pass
            elif context[varname] is not globaldict[varname]:
                raise ValueError("Variable conflict between __main__.%s and %s.%s" % (varname,callable.__module__,varname))
            pass
        else:
            # assign variable
            context[varname]=globaldict[varname]
            pass
        pass
    
    
    return scriptified

    
