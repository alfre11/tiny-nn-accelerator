`timescale 1ns/1ps

module nn_accelerator_tb;

    logic clk;
    logic rst;
    logic start;
    logic done;
    logic [3:0] prediction;

    logic signed [31:0] expected_prediction [0:0];

    nn_accelerator dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .done(done),
        .prediction(prediction)
    );

    // Clock: 10 ns period
    always #5 clk = ~clk;

    initial begin
        $readmemh("data/mem/expected_prediction.mem", expected_prediction);

        clk = 0;
        rst = 1;
        start = 0;

        // Hold reset for a few cycles
        repeat (5) @(posedge clk);
        rst = 0;

        // Pulse start
        @(posedge clk);
        start = 1;

        @(posedge clk);
        start = 0;

        // Wait for accelerator to finish
        wait(done == 1);

        $display("Prediction:          %0d", prediction);
        $display("Expected prediction: %0d", expected_prediction[0]);

        if (prediction == expected_prediction[0][3:0]) begin
            $display("PASS: prediction matches");
        end else begin
            $display("FAIL: prediction mismatch");
        end

        $finish;
    end

endmodule