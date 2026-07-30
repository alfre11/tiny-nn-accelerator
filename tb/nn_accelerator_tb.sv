`timescale 1ns/1ps

module nn_accelerator_tb;

    logic clk;
    logic rst;
    logic start;
    logic done;
    logic [3:0] prediction;

    logic signed [31:0] expected_prediction [0:0];
    logic signed [31:0] expected_output [0:9];

    nn_accelerator dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .done(done),
        .prediction(prediction)
    );

    always #5 clk = ~clk;

    initial begin
        $readmemh("data/mem/expected_prediction.mem", expected_prediction);
        $readmemh("data/mem/expected_output.mem", expected_output);

        clk = 1'b0;
        rst = 1'b1;
        start = 1'b0;

        // Hold reset for a few cycles
        repeat (5) @(posedge clk);

        // Deassert reset away from the active clock edge
        @(negedge clk);
        rst = 1'b0;

        // Wait a couple cycles
        repeat (2) @(posedge clk);

        // Pulse start, also away from active clock edge
        @(negedge clk);
        start = 1'b1;

        @(negedge clk);
        start = 1'b0;

        fork
            begin
                wait(done == 1'b1);
            end

            begin
                repeat (5000) @(posedge clk);
                $display("TIMEOUT: accelerator never asserted done");
                $display("Current state = %0d", dut.state);
                $display("input_idx = %0d", dut.input_idx);
                $display("hidden_idx = %0d", dut.hidden_idx);
                $display("output_idx = %0d", dut.output_idx);
                $display("l2_hidden_idx = %0d", dut.l2_hidden_idx);
                $display("argmax_idx = %0d", dut.argmax_idx);
                $finish;
            end
        join_any

        disable fork;

        $display("Prediction:          %0d", prediction);
        $display("Expected prediction: %0d", expected_prediction[0]);

        if (prediction == expected_prediction[0][3:0]) begin
            $display("PASS: prediction matches");
        end else begin
            $display("FAIL: prediction mismatch");
        end

        $display("");
        $display("Output logits:");

        for (int i = 0; i < 10; i++) begin
            $display(
                "output_mem[%0d] = %0d, expected = %0d",
                i,
                dut.output_mem[i],
                expected_output[i]
            );
        end

        $finish;
    end

endmodule