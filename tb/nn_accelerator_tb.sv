`timescale 1ns/1ps

module nn_accelerator_tb;

    logic clk;
    logic rst;
    logic start;
    logic done;
    logic [3:0] prediction;

    logic signed [31:0] expected_prediction [0:0];
    logic signed [31:0] expected_output [0:9];
    logic signed [31:0] expected_hidden [0:15];

    int output_errors;
    int hidden_errors;

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
        $readmemh("data/mem/expected_hidden.mem", expected_hidden);

        clk = 1'b0;
        rst = 1'b1;
        start = 1'b0;
        output_errors = 0;
        hidden_errors = 0;

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

        $display("");
        $display("Prediction check:");
        $display("Prediction:          %0d", prediction);
        $display("Expected prediction: %0d", expected_prediction[0]);

        if (prediction == expected_prediction[0][3:0]) begin
            $display("PASS: prediction matches");
        end else begin
            $display("FAIL: prediction mismatch");
        end

        $display("");
        $display("Hidden values:");

        for (int i = 0; i < 16; i++) begin
            $display(
                "hidden_mem[%0d] = %0d, expected = %0d",
                i,
                dut.hidden_mem[i],
                expected_hidden[i]
            );

            if (dut.hidden_mem[i] !== expected_hidden[i]) begin
                hidden_errors++;
            end
        end

        if (hidden_errors == 0) begin
            $display("PASS: all hidden values match");
        end else begin
            $display("FAIL: %0d hidden values mismatched", hidden_errors);
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

            if (dut.output_mem[i] !== expected_output[i]) begin
                output_errors++;
            end
        end

        if (output_errors == 0) begin
            $display("PASS: all output logits match");
        end else begin
            $display("FAIL: %0d output logits mismatched", output_errors);
        end

        $display("");

        if (
            prediction == expected_prediction[0][3:0] &&
            hidden_errors == 0 &&
            output_errors == 0
        ) begin
            $display("OVERALL RESULT: PASS");
        end else begin
            $display("OVERALL RESULT: FAIL");
        end

        $finish;
    end

endmodule