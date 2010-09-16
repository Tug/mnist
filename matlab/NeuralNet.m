classdef NeuralNet < handle
    %NEURALNET with backpropagation
    
    properties (GetAccess='public', SetAccess='public')
        % w is a cell array containing in each cell the weights of a layer
        % in the network. So, each cell contains a 2D matrix where lines
        % represent the id of the neurons of the current layer and the
        % column represent the id of the neurons from the previous layer.
        w;
        Dw;
        % eta is the coefficient of learning
        eta = 0.1;
        % m is the momentum (set it to 0 to desactivate)
        m = 0.95;
        % g is a cell array of function containing for each layer, the
        % corresponding activation function
        g;
        dg;
        % g_out is the activation function of the last layer
        g_out = @(x)( 1./(1+exp(-x)) );
        % g_hidden is the activation function of the hidden layers
        g_hidden = @(x)( 1./(1+exp(-x+5)) + 1./(1+exp(-x-5)) - 1 );
        % dg_out is the derivative of g_out
        dg_out = @(x)( exp(-x)./(1+exp(-x)).^2 );
        % dg_hidden is the derivative of g_hidden
        dg_hidden = @(x)( exp(-x+5)./(1+exp(-x+5)).^2 ...
                        + exp(-x-5)./(1+exp(-x-5)).^2 );
        % S is a vector defining the number of neurons for each layers
        S;
        %nbLayers is the length of S
        nbLayers;
    end

    methods
        function this = NeuralNet(S)
            this.S = S;
            this.nbLayers = length(S);
            this.w = cell(this.nbLayers-1,1);
            %Dw is the same size as w
            this.Dw = cell(this.nbLayers-1,1);
            for i=1:this.nbLayers-1,
                % weights in [-2, 2]
                this.w{i} = 2*(2*(rand(S(i+1),S(i)+1)-1));
                this.Dw{i} = zeros(S(i+1),S(i)+1);
            end
            this.g = cell(this.nbLayers-1,1);
            for i=1:this.nbLayers-2,
                this.g{i} = this.g_hidden;
                this.dg{i} = this.dg_hidden;
            end
            this.g{this.nbLayers-1} = this.g_out;
            this.dg{this.nbLayers-1} = this.dg_out;
        end
        
        function E = BackProp(this, xin, tout)
            h = cell(this.nbLayers-1,1);
            % Init x
            x = cell(this.nbLayers,1);
            % insert xin in the first layer of x
            x{1} = [xin; 1];
            for n=2:this.nbLayers-1,
                x{n} = [zeros(this.S(n),1); 1];
            end
            x{this.nbLayers} = zeros(this.S(this.nbLayers),1);
            % Propagate
            for n=1:this.nbLayers-1,
                h{n} = this.w{n} * x{n};
                x{n+1}(1:this.S(n+1)) = this.g{n}( h{n} );
            end
            out = this.nbLayers-1;
            % compute the error of the ouput :
            % the answer of the supervisor - the output
            % times g'(h(out))
            % input layer is x{1} not 0 so last one is x{out+1}
            d = this.dg{out}(h{out}) .* (tout - x{out+1});
            % the derivative of g_out is : g_out * (1 - g_out)
            % thus we can optimize:
            %d = x{out+1} .* (1 - x{out+1}) .* (tout - x{out+1});
            this.Dw{out} = this.eta .* d * x{out}';
            % retropropagate the error
            for n=out:-1:2,
                % d is the error at layer n : d = g'(h) .* w'.d
                % w'.d :
                % w11 w21  .  d1   =   w11*d1 + w21*d2
                % w12 w22     d2       w12*d1 + w22*d2
                d = this.dg{n-1}(h{n-1}) .* (this.w{n}(:,1:end-1)' * d);
                % Dw is a matrix obtain by the vector product (times eta) of:
                % -> d: error of the right layer (without the biasis)
                % -> x: output of the left layer (transposed to be a column vector)
                % d1  .  x1 x2 x3  =  d1*x1 d1*x2 d1*x3
                % d2                  d2*x1 d2*x2 d2*x3
                this.Dw{n-1} = this.eta .* d * x{n-1}' + this.m .* this.Dw{n-1};
            end
            % weights update
            for n=1:out,
                this.w{n} = this.w{n} + this.Dw{n};
            end
            E = 0.5 * sum( (tout - x{out+1}).^2 );
        end
        
        function E = learn(this, xins, touts)
            s = size(xins);
            E = 0;
            for i=1:s(1),
                E = E + this.BackProp(xins(i,:)', touts(i,:)');
            end
        end
        
        function [Etrue, Eestim] = evaluateError(this, xins, touts)
            Etrue = 0;
            dEdwTrue = cell(this.nbLayers-1,1);
            for i=1:this.nbLayers-1,
                dEdwTrue{i} = zeros(this.S(i+1),this.S(i)+1);
            end
            for i=1:s(1),
                Etrue = Etrue + this.BackProp(xins(i,:)', touts(i,:)');
                for j=1:length(dEdwTrue),
                    dEdwTrue{i} = dEdwTrue{i} + this.Dw{i}./this.eta;
                end
            end
            
            E2 = this.w; %same size
            dEdwEstim = this.w;
            epsi = 0.00001;
            wbak = this.w;
            for n=1:this.nbLayers-1,
                s = size(this.w{n});
                for lin=1:s(1),
                    for col=1:s(2),
                        this.w{n}(lin,col) = this.w{n}(lin,col) + epsi;
                        E2{n}(lin,col) = this.learn(xins, touts);
                        this.w = wbak;
                    end
                end
                dEdwEstim{n} = (Etrue - E2{n}) ./ epsi;
            end
            Eestim = dEdw;
        end
        
        function xout = test(this, xin)
            % insert xin
            x = [xin; 1];
            % Propagate x
            for n=1:this.nbLayers-2,
                x = [this.g{n}(this.w{n} * x); 1];
            end
            out = this.nbLayers-1;
            xout = this.g{out}(this.w{out} * x);
        end
        
    end
end

